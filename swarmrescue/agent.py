"""Autonomous search-and-rescue robot.

Each :class:`Robot` owns two private maps:

* ``known`` is its belief of the disaster zone (-1 unknown, 0 free,
  1 debris). Its short-range debris sensor updates it, and so do radio
  merges with teammates.
* ``trail`` is its :class:`PheromoneTrail`: the ant-style marks recording
  which cells were searched, and how recently and how often.

A robot never sees the ground-truth grid. It decides every move from these
maps plus the teammates it can perceive right now. There is no central
controller, so there is no single point of failure.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from swarmrescue.config import SwarmConfig
from swarmrescue.learning import AdaptiveWeightLearner
from swarmrescue.world import DEBRIS, FREE, Cell, manhattan, neighbors4

UNKNOWN: int = -1
DEADLOCK_TICKS: int = 3
# With evaporation on, a searched cell whose pheromone has faded below this
# level counts as stale, and idle robots patrol back to it.
STALE_PHEROMONE: float = 0.05

MODE_PHEROMONE = "pheromone"
MODE_BFS = "reroute"
MODE_DEADLOCK = "deadlock"
MODE_BLOCKED = "blocked"
MODE_IDLE = "idle"
MODE_PATROL = "patrol"
MODE_PING = "ping"
MODE_OFF = "off"


class PheromoneTrail:
    """Digital ant pheromone: where and how intensely a robot has searched.

    Attributes:
        intensity: Float map of pheromone strength. Each visit deposits 1.0.
        visited: Boolean map: has this cell ever been searched (by this robot
            or by a teammate whose trail was merged over the radio link)?
    """

    def __init__(self, size: int) -> None:
        """Create an empty trail for a ``size`` x ``size`` zone."""
        self.intensity: np.ndarray = np.zeros((size, size), dtype=np.float64)
        self.visited: np.ndarray = np.zeros((size, size), dtype=bool)

    def deposit(self, cell: Cell, amount: float = 1.0) -> None:
        """Lay pheromone on ``cell`` (the robot searched it)."""
        self.intensity[cell] += amount
        self.visited[cell] = True

    def strength(self, cell: Cell) -> float:
        """Pheromone intensity at ``cell``."""
        return float(self.intensity[cell])

    def evaporate(self, rate: float) -> None:
        """Let the pheromone fade: ``intensity *= (1 - rate)``.

        Like real ant trails, old marks weaken, so areas searched long ago
        attract robots again. The ``visited`` memory is not erased.
        """
        if rate > 0.0:
            self.intensity *= 1.0 - rate

    def stale_mask(self, threshold: float = STALE_PHEROMONE) -> np.ndarray:
        """Searched cells whose pheromone has faded below ``threshold``."""
        return self.visited & (self.intensity < threshold)

    def is_visited(self, cell: Cell) -> bool:
        """True if ``cell`` has been searched according to this trail."""
        return bool(self.visited[cell])

    def merge(self, other: PheromoneTrail) -> bool:
        """Combine with a teammate's trail (element-wise maximum).

        Returns:
            True if the set of searched cells grew (routes planned on the old
            trail may no longer lead to an unsearched cell).
        """
        np.maximum(self.intensity, other.intensity, out=self.intensity)
        grew = bool((other.visited & ~self.visited).any())
        self.visited |= other.visited
        return grew

    def copy(self) -> PheromoneTrail:
        """Independent deep copy (used for order-independent radio merges)."""
        clone = PheromoneTrail(self.intensity.shape[0])
        clone.intensity = self.intensity.copy()
        clone.visited = self.visited.copy()
        return clone


def bfs_path(
    passable: np.ndarray,
    start: Cell,
    goal_mask: np.ndarray,
    blocked: Iterable[Cell] = (),
) -> list[Cell] | None:
    """Breadth-first search to the nearest goal cell (used to reroute).

    Args:
        passable: Boolean grid, True where the robot may drive.
        start: Starting cell (always expanded, even if not passable).
        goal_mask: Boolean grid, True at acceptable destination cells.
        blocked: Extra cells treated as obstacles (e.g. other robots).

    Returns:
        The shortest 4-connected path ``[start, ..., goal]`` to the closest
        goal cell other than ``start``, or ``None`` if no goal is reachable.
    """
    size = passable.shape[0]
    ok = passable.tolist()
    goals = goal_mask.tolist()
    for r, c in blocked:
        if 0 <= r < size and 0 <= c < size:
            ok[r][c] = False
    parent: dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])
    while queue:
        cell = queue.popleft()
        for nb in neighbors4(cell, size):
            if nb in parent or not ok[nb[0]][nb[1]]:
                continue
            parent[nb] = cell
            if goals[nb[0]][nb[1]]:
                return _trace_path(parent, nb)
            queue.append(nb)
    return None


def _trace_path(parent: dict[Cell, Cell | None], goal: Cell) -> list[Cell]:
    """Follow BFS parent links back from ``goal``; returns ``[start, ..., goal]``."""
    path = [goal]
    prev = parent[goal]
    while prev is not None:
        path.append(prev)
        prev = parent[prev]
    return path[::-1]


@dataclass(frozen=True)
class MoveContext:
    """Everything a robot perceives when deciding one move."""

    cfg: SwarmConfig
    blocked: set[Cell]
    others: list[Cell]
    rng: np.random.Generator
    pings: tuple[Cell, ...]
    free_nbs: list[Cell]
    options: list[Cell]
    reachable: list[Cell]


Decision = tuple[str, Cell | None]


def crowding(cell: Cell, others: Iterable[Cell]) -> float:
    """Crowding penalty: sum of ``1 / (1 + manhattan)`` to perceived robots."""
    return float(sum(1.0 / (1.0 + manhattan(cell, o)) for o in others))


def crowding_many(cells: Sequence[Cell], others: Sequence[Cell]) -> np.ndarray:
    """Vectorised :func:`crowding` for several candidate cells at once (O(C x R) in NumPy)."""
    if not others:
        return np.zeros(len(cells))
    dist = np.abs(np.asarray(cells)[:, None, :] - np.asarray(others)[None, :, :]).sum(axis=2)
    result: np.ndarray = (1.0 / (1.0 + dist)).sum(axis=1)
    return result


class Robot:
    """A decentralized search-and-rescue robot.

    Attributes:
        robot_id: Integer identifier (robot 0 fails in the Round 2 aftershock).
        pos: Current cell.
        battery: Remaining energy, counted in moves.
        alive: False once the robot has suffered a hardware failure.
        trail: Private :class:`PheromoneTrail`.
        known: Private occupancy belief (-1 unknown, 0 free, 1 debris).
        stuck_ticks: Consecutive ticks the robot wanted to move but could not.
        moves: Total moves made (energy used).
        mode: How the last decision was made (for telemetry and the dashboard).
        belief_version: Bumped whenever ``known`` or the searched set changes;
            a cached BFS route is reused only while the version is unchanged.
        bfs_calls: BFS searches actually run (efficiency telemetry).
        route_cache_hits: Moves served from a still-valid cached route.
    """

    def __init__(self, robot_id: int, start: Cell, grid_size: int, battery: int) -> None:
        """Deploy a robot at ``start`` with empty knowledge and a full battery."""
        self.robot_id: int = robot_id
        self.pos: Cell = start
        self.battery: int = battery
        self.alive: bool = True
        self.trail: PheromoneTrail = PheromoneTrail(grid_size)
        self.trail.deposit(start)
        self.known: np.ndarray = np.full((grid_size, grid_size), UNKNOWN, dtype=np.int8)
        self.known[start] = FREE
        self.stuck_ticks: int = 0
        self.moves: int = 0
        self.mode: str = MODE_IDLE
        self.belief_version: int = 0
        self.bfs_calls: int = 0
        self.route_cache_hits: int = 0
        self._route: list[Cell] = []
        self._route_version: int = -1
        self.last_surprises: int = 0
        self.learner: AdaptiveWeightLearner | None = None  # online learning (optional)

    @property
    def active(self) -> bool:
        """True if the robot can still move (alive with battery left)."""
        return self.alive and self.battery > 0

    def fail(self) -> None:
        """Permanently disable the robot. It stays in place as an obstacle."""
        self.alive = False
        self.mode = MODE_OFF

    def sense_debris(self, grid: np.ndarray, sense_range: int) -> int:
        """Scan the ground truth within a square window of ``sense_range``.

        Args:
            grid: True occupancy grid of the disaster zone.
            sense_range: Chebyshev sensing radius.

        Returns:
            Number of debris cells discovered where the robot believed there
            was free space, i.e. aftershock debris on a known route, which
            forces a reroute.
        """
        if not self.alive:
            return 0
        size = grid.shape[0]
        r, c = self.pos
        r0, r1 = max(0, r - sense_range), min(size, r + sense_range + 1)
        c0, c1 = max(0, c - sense_range), min(size, c + sense_range + 1)
        before = self.known[r0:r1, c0:c1]
        truth = grid[r0:r1, c0:c1]
        surprises = int(((before == FREE) & (truth == DEBRIS)).sum())
        self.last_surprises = surprises
        if surprises:
            self.recalibrate_route()
        if not np.array_equal(before, truth):
            self.belief_version += 1
        self.known[r0:r1, c0:c1] = truth
        return surprises

    def recalibrate_route(self) -> None:
        """Recalibrate route: drop the cached trajectory so the next decision replans it with BFS.

        Called the moment debris is sensed on a route the robot believed free;
        replanning happens in the same tick, well inside the real-time budget.
        """
        self._route = []
        self._route_version = -1

    def absorb(self, trail: PheromoneTrail, known: np.ndarray) -> None:
        """Merge a teammate's trail and debris map (received over the radio link)."""
        grew = self.trail.merge(trail)
        merged = np.maximum(self.known, known)
        if grew or not np.array_equal(merged, self.known):
            self.belief_version += 1
            self.known = merged

    def score(self, cell: Cell, others: list[Cell], cfg: SwarmConfig, rng: np.random.Generator) -> float:
        """Ant-pheromone move score (lower is better).

        ``score = W_p * pheromone + W_s * crowding + R * U(0, 1)``
        """
        return (
            cfg.pheromone_weight * self.trail.strength(cell)
            + cfg.spread_weight * crowding(cell, others)
            + cfg.randomness * float(rng.random())
        )

    def _reroute(self, goals: np.ndarray, blocked: set[Cell], passable: np.ndarray | None = None) -> Cell | None:
        """First step of the shortest BFS route to the nearest goal, or None (the route is cached)."""
        if not goals.any():
            return None
        grid = self.known == FREE if passable is None else passable
        self.bfs_calls += 1
        path = bfs_path(grid, self.pos, goals, blocked)
        self._route = [] if path is None else path[1:]
        self._route_version = self.belief_version
        return None if path is None else path[1]

    def _cached_step(self, goals: np.ndarray, blocked: set[Cell]) -> Cell | None:
        """Route cache: next step of the cached route (trajectory) if it is still valid, else None.

        Valid means: the robot's belief has not changed since planning (no new
        debris, no merged trail), the route's goal is still a goal, and no
        perceived teammate stands on the remaining route. Under these conditions
        the rest of a shortest route is still a shortest route, so skipping BFS
        is safe (sub-path optimality).
        """
        route = self._route
        if not route or self._route_version != self.belief_version or not goals[route[-1]]:
            return None
        if any(cell in blocked for cell in route):
            return None
        self.route_cache_hits += 1
        return route[0]

    def home_in_on_ping(self, pings: Sequence[Cell], blocked: set[Cell]) -> Cell | None:
        """First step toward the nearest heard survivor ping.

        Sound travels through rubble, so the robot knows where the survivor is
        but not the route. It plans optimistically through unknown cells and
        avoids known debris. Its sensor corrects the plan as it moves, and it
        reroutes when it finds new debris.

        Returns:
            A known-free neighbouring cell on the shortest optimistic route, or
            ``None`` if no ping is heard or no route exists.
        """
        if not pings:
            return None
        size = self.known.shape[0]
        goals = np.zeros((size, size), dtype=bool)
        for cell in pings:
            goals[cell] = True
        step = self._reroute(goals, blocked, self.known != DEBRIS)
        if step is None or self.known[step] != FREE:
            return None
        return step

    def choose_move(
        self,
        cfg: SwarmConfig,
        blocked: set[Cell],
        others: list[Cell],
        rng: np.random.Generator,
        pings: Sequence[Cell] = (),
    ) -> Cell | None:
        """Decide this tick's move using only local knowledge.

        Behaviours are tried in priority order and the first that decides wins:
        off, deadlock breaker, survivor ping, pheromone rule (while an unvisited
        neighbour exists), BFS reroute to the nearest unvisited known-free cell,
        patrol back to a stale cell (evaporation only), otherwise idle to save
        battery.

        Args:
            cfg: Mission configuration (behaviour weights).
            blocked: Cells the robot perceives as occupied or reserved.
            others: Positions of perceived working teammates, for crowding.
            rng: Seeded random generator.
            pings: Survivor cells whose acoustic ping the robot hears this tick.

        Returns:
            The neighbouring cell to move into, or ``None`` to stay put.
        """
        free_nbs = [nb for nb in neighbors4(self.pos, self.known.shape[0]) if self.known[nb] == FREE]
        reachable = [n for n in free_nbs if n not in blocked]
        safe = [n for n in reachable if self._keeps_clearance(n, blocked, cfg.safety_margin)]
        ctx = MoveContext(cfg, blocked, others, rng, tuple(pings), free_nbs, safe, reachable)
        for behaviour in self._behaviours():
            decision = behaviour(ctx)
            if decision is not None:
                self.mode, move = self._enforce_margin(decision, ctx)
                return move
        self.mode = MODE_IDLE  # nothing left to search in my belief: save energy
        return None

    def _keeps_clearance(self, cell: Cell, robots: set[Cell], margin: int) -> bool:
        """Safety margin: moving to ``cell`` keeps clearance > margin, or does not reduce it."""
        if margin <= 0 or not robots:
            return True
        after = min(manhattan(cell, r) for r in robots)
        return after > margin or after >= min(manhattan(self.pos, r) for r in robots)

    def _enforce_margin(self, decision: Decision, ctx: MoveContext) -> Decision:
        """Planned steps (ping, reroute, patrol) that would break the safety margin become a wait."""
        mode, move = decision
        if move is None or mode == MODE_DEADLOCK or move in ctx.options:
            return decision
        return MODE_BLOCKED, None

    def _behaviours(self) -> tuple[Callable[[MoveContext], Decision | None], ...]:
        """Behaviours in priority order (see :meth:`choose_move`)."""
        return (
            self._switched_off,
            self._break_deadlock,
            self._follow_ping,
            self._pheromone_rule,
            self._reroute_to_frontier,
            self._patrol_stale_area,
        )

    def _switched_off(self, ctx: MoveContext) -> Decision | None:
        """A failed or flat robot never moves."""
        return None if self.active else (MODE_OFF, None)

    def _break_deadlock(self, ctx: MoveContext) -> Decision | None:
        """Deadlock breaker: after DEADLOCK_TICKS blocked ticks, take a random free neighbour.

        The breaker may relax the safety margin (never the no-shared-cell rule) so
        the swarm can never freeze.
        """
        choices = ctx.options or ctx.reachable
        if self.stuck_ticks < DEADLOCK_TICKS or not choices:
            return None
        return MODE_DEADLOCK, choices[int(ctx.rng.integers(len(choices)))]

    def _follow_ping(self, ctx: MoveContext) -> Decision | None:
        """Head for a survivor whose acoustic ping is heard."""
        step = self.home_in_on_ping(ctx.pings, ctx.blocked)
        return None if step is None else (MODE_PING, step)

    def _pheromone_rule(self, ctx: MoveContext) -> Decision | None:
        """Ant rule: lowest pheromone + crowding + noise among free neighbours."""
        if all(self.trail.is_visited(nb) for nb in ctx.free_nbs):
            return None
        if not ctx.options:
            return MODE_BLOCKED, None
        opts = np.asarray(ctx.options)
        cfg = ctx.cfg
        pheromone_w, spread_w = self.behaviour_weights(cfg)
        scores = (
            pheromone_w * self.trail.intensity[opts[:, 0], opts[:, 1]]
            + spread_w * crowding_many(ctx.options, ctx.others)
            + cfg.randomness * ctx.rng.random(len(ctx.options))
        )
        return MODE_PHEROMONE, ctx.options[int(np.argmin(scores))]

    def _reroute_to_frontier(self, ctx: MoveContext) -> Decision | None:
        """All neighbours searched: BFS reroute to the nearest unsearched known-free cell."""
        passable = self.known == FREE
        frontier = passable & ~self.trail.visited
        if not frontier.any():
            return None
        step = self._cached_step(frontier, ctx.blocked) or self._reroute(frontier, ctx.blocked, passable)
        return (MODE_BFS, step) if step is not None else (MODE_BLOCKED, None)  # blocked: a teammate is in the way

    def _patrol_stale_area(self, ctx: MoveContext) -> Decision | None:
        """With evaporation, re-sweep the nearest area searched long ago."""
        if not ctx.cfg.use_evaporation:
            return None
        passable = self.known == FREE
        step = self._reroute(passable & self.trail.stale_mask(), ctx.blocked, passable)
        return None if step is None else (MODE_PATROL, step)

    def behaviour_weights(self, cfg: SwarmConfig) -> tuple[float, float]:
        """``(pheromone_weight, spread_weight)``: the online learner's current preset, else the config."""
        return self.learner.weights if self.learner is not None else (cfg.pheromone_weight, cfg.spread_weight)

    def _learn(self, move: Cell | None) -> None:
        """Online learning reward: did this tick's move search a new cell (energy efficiency)?"""
        if self.learner is not None and self.active:
            self.learner.observe(moved=move is not None, new_cell=move is not None and not self.trail.is_visited(move))

    def commit(self, move: Cell | None) -> None:
        """Apply the outcome of this tick's decision.

        Args:
            move: Cell moved into, or ``None`` if the robot stayed put.
        """
        self._learn(move)
        if move is None:
            if self.mode == MODE_BLOCKED:
                self.stuck_ticks += 1
            elif self.mode in (MODE_IDLE, MODE_OFF):
                self.stuck_ticks = 0
            return
        if self._route and self._route[0] == move:
            self._route.pop(0)
        else:
            self._route = []
        self.pos = move
        self.battery -= 1
        self.moves += 1
        self.trail.deposit(move)
        self.stuck_ticks = 0
