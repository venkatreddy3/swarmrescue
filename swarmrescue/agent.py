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
from typing import Iterable

import numpy as np

from swarmrescue.config import SwarmConfig
from swarmrescue.world import DEBRIS, FREE, Cell, manhattan, neighbors4

UNKNOWN: int = -1
DEADLOCK_TICKS: int = 3

MODE_PHEROMONE = "pheromone"
MODE_BFS = "reroute"
MODE_DEADLOCK = "deadlock"
MODE_BLOCKED = "blocked"
MODE_IDLE = "idle"
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

    def is_visited(self, cell: Cell) -> bool:
        """True if ``cell`` has been searched according to this trail."""
        return bool(self.visited[cell])

    def merge(self, other: "PheromoneTrail") -> None:
        """Combine with a teammate's trail (element-wise maximum)."""
        np.maximum(self.intensity, other.intensity, out=self.intensity)
        self.visited |= other.visited

    def copy(self) -> "PheromoneTrail":
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
                path = [nb]
                prev = parent[nb]
                while prev is not None:
                    path.append(prev)
                    prev = parent[prev]
                return path[::-1]
            queue.append(nb)
    return None


def crowding(cell: Cell, others: Iterable[Cell]) -> float:
    """Crowding penalty: sum of ``1 / (1 + manhattan)`` to perceived robots."""
    return float(sum(1.0 / (1.0 + manhattan(cell, o)) for o in others))


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
        self.known[r0:r1, c0:c1] = truth
        return surprises

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
        """First step of the shortest BFS route to the nearest goal, or None."""
        if not goals.any():
            return None
        grid = self.known == FREE if passable is None else passable
        path = bfs_path(grid, self.pos, goals, blocked)
        return None if path is None else path[1]

    def choose_move(
        self,
        cfg: SwarmConfig,
        blocked: set[Cell],
        others: list[Cell],
        rng: np.random.Generator,
    ) -> Cell | None:
        """Decide this tick's move using only local knowledge.

        Priority: deadlock breaker, then the pheromone rule (while an unvisited
        neighbour exists), then a BFS reroute to the nearest unvisited
        known-free cell, then idle to save battery.

        Args:
            cfg: Mission configuration (behaviour weights).
            blocked: Cells the robot perceives as occupied or reserved.
            others: Positions of perceived working teammates, for crowding.
            rng: Seeded random generator.

        Returns:
            The neighbouring cell to move into, or ``None`` to stay put.
        """
        if not self.active:
            self.mode = MODE_OFF
            return None
        size = self.known.shape[0]
        free_nbs = [nb for nb in neighbors4(self.pos, size) if self.known[nb] == FREE]
        options = [nb for nb in free_nbs if nb not in blocked]

        if self.stuck_ticks >= DEADLOCK_TICKS and options:
            self.mode = MODE_DEADLOCK
            return options[int(rng.integers(len(options)))]

        if any(not self.trail.is_visited(nb) for nb in free_nbs):
            if not options:
                self.mode = MODE_BLOCKED
                return None
            self.mode = MODE_PHEROMONE
            scores = [self.score(nb, others, cfg, rng) for nb in options]
            return options[int(np.argmin(scores))]

        # All neighbours already searched: reroute to the nearest unsearched cell.
        passable = self.known == FREE
        frontier = passable & ~self.trail.visited
        if not frontier.any():
            self.mode = MODE_IDLE  # nothing left to search in my belief: save energy
            return None
        step = self._reroute(frontier, blocked, passable)
        if step is not None:
            self.mode = MODE_BFS
            return step
        self.mode = MODE_BLOCKED  # a teammate is in the way; wait
        return None

    def commit(self, move: Cell | None) -> None:
        """Apply the outcome of this tick's decision.

        Args:
            move: Cell moved into, or ``None`` if the robot stayed put.
        """
        if move is None:
            if self.mode == MODE_BLOCKED:
                self.stuck_ticks += 1
            elif self.mode in (MODE_IDLE, MODE_OFF):
                self.stuck_ticks = 0
            return
        self.pos = move
        self.battery -= 1
        self.moves += 1
        self.trail.deposit(move)
        self.stuck_ticks = 0
