"""Tick-based mission simulator and fitness function.

One tick = every robot senses, (optionally) shares maps with nearby
teammates, then robots move one after another. Sequential moves plus a
per-robot occupancy check make the swarm collision-free by construction; the
simulator still audits every tick and counts any collision it finds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from swarmrescue.agent import MODE_BLOCKED, MODE_IDLE, Agent
from swarmrescue.config import SwarmConfig
from swarmrescue.coordination import perceive, perception_radius, share_maps
from swarmrescue.world import FREE, WALL, Cell, World, apply_shift, generate_world, reachable

COVERAGE_TARGET: float = 0.9
FAILED_ROBOT_ID: int = 0


@dataclass(frozen=True)
class Frame:
    """Snapshot of the mission after one tick (used for animation).

    Attributes:
        tick: Tick number (0 = initial state).
        grid: Ground-truth occupancy grid at this tick.
        visited: Cells visited by any robot so far.
        positions: Robot positions, indexed by robot id.
        alive: Whether each robot is still operational.
        found: Whether each survivor has been found.
        coverage: Coverage fraction at this tick.
    """

    tick: int
    grid: np.ndarray
    visited: np.ndarray
    positions: tuple[Cell, ...]
    alive: tuple[bool, ...]
    found: tuple[bool, ...]
    coverage: float


@dataclass(frozen=True)
class SimulationResult:
    """Outcome and metrics of one simulated mission."""

    round_no: int
    seed: int
    coverage: float
    survivors_found: int
    survivors_total: int
    tick_at_90: int | None
    ticks_run: int
    energy_moves: int
    collisions: int
    fitness: float
    max_latency_ms: float
    mean_latency_ms: float
    reachable_cells: int
    visited_cells: int
    wall_surprises: int
    interlock_trips: int
    depleted_robots: int
    final_grid: np.ndarray
    visited_mask: np.ndarray
    final_positions: tuple[Cell, ...]
    alive: tuple[bool, ...]
    survivors: tuple[Cell, ...]
    found: tuple[bool, ...]
    coverage_curve: tuple[float, ...]
    frames: tuple[Frame, ...] = field(default_factory=tuple)

    @property
    def survivors_ratio(self) -> float:
        """Fraction of survivors found (1.0 if there were none)."""
        return self.survivors_found / self.survivors_total if self.survivors_total else 1.0

    def summary(self) -> dict[str, float | int | None]:
        """Key scalar metrics as a flat dictionary."""
        return {
            "round": self.round_no,
            "seed": self.seed,
            "coverage": round(self.coverage, 4),
            "survivors_found": self.survivors_found,
            "survivors_total": self.survivors_total,
            "tick_at_90": self.tick_at_90,
            "ticks_run": self.ticks_run,
            "energy_moves": self.energy_moves,
            "collisions": self.collisions,
            "fitness": round(self.fitness, 3),
            "max_latency_ms": round(self.max_latency_ms, 3),
            "wall_surprises": self.wall_surprises,
        }


def compute_fitness(
    coverage: float,
    survivors_ratio: float,
    tick_at_90: int | None,
    max_ticks: int,
    energy_moves: int,
    collisions: int,
) -> float:
    """Mission fitness (higher is better).

    ``100*coverage + 20*survivors_ratio + 20*speed - 0.01*energy - 100*collisions``
    where ``speed = 1 - tick_at_90 / max_ticks`` (0 if 90% was never reached).
    """
    speed = 0.0 if tick_at_90 is None else 1.0 - tick_at_90 / max_ticks
    return (
        100.0 * coverage
        + 20.0 * survivors_ratio
        + 20.0 * speed
        - 0.01 * energy_moves
        - 100.0 * collisions
    )


def compute_coverage(visited: np.ndarray, reachable_mask: np.ndarray) -> float:
    """Fraction of reachable free cells that have been visited, in [0, 1]."""
    total = int(reachable_mask.sum())
    if total == 0:
        return 1.0
    return int((visited & reachable_mask).sum()) / total


def count_collisions(
    agents: list[Agent], grid: np.ndarray, previous: list[Cell]
) -> int:
    """Audit one tick: shared cells, robots inside walls and head-on swaps."""
    positions = [a.pos for a in agents]
    collisions = len(positions) - len(set(positions))
    collisions += sum(1 for p in positions if grid[p] == WALL)
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if positions[i] == previous[j] and positions[j] == previous[i] and positions[i] != previous[i]:
                collisions += 1
    return collisions


def post_shift_reachable(grid: np.ndarray, agents: list[Agent], visited: np.ndarray) -> np.ndarray:
    """Recompute the coverage denominator after the Round 2 shift.

    Reachable = free cells connected to a working robot (failed robots are
    obstacles) plus already-visited cells that are still free, since that
    exploration work was genuinely done.
    """
    blocked = grid.copy()
    for a in agents:
        if not a.alive:
            blocked[a.pos] = WALL
    mask = reachable(blocked, [a.pos for a in agents if a.alive])
    return mask | (visited & (grid == FREE))


def _snapshot(tick: int, world: World, visited: np.ndarray, agents: list[Agent], found: list[bool], coverage: float) -> Frame:
    """Build an immutable :class:`Frame` of the current state."""
    return Frame(
        tick=tick,
        grid=world.grid.copy(),
        visited=visited.copy(),
        positions=tuple(a.pos for a in agents),
        alive=tuple(a.alive for a in agents),
        found=tuple(found),
        coverage=coverage,
    )


def simulate(cfg: SwarmConfig, round_no: int = 1, record_frames: bool = False) -> SimulationResult:
    """Run one full rescue mission.

    Args:
        cfg: Mission configuration (``cfg.seed`` drives all randomness).
        round_no: 1 = static building; 2 = scenario shift at ``cfg.shift_tick``
            (new debris, robot 0 fails, radio link lost).
        record_frames: Store a :class:`Frame` per tick for animation.

    Returns:
        A :class:`SimulationResult` with all metrics.

    Raises:
        ValueError: If ``round_no`` is not 1 or 2.
    """
    if round_no not in (1, 2):
        raise ValueError("round_no must be 1 or 2")
    world_ss, agent_ss, shift_ss = np.random.SeedSequence(cfg.seed).spawn(3)
    world = generate_world(cfg, np.random.default_rng(world_ss))
    agent_rng = np.random.default_rng(agent_ss)
    shift_rng = np.random.default_rng(shift_ss)

    n = cfg.grid_size
    agents = [Agent(i, start, n, cfg.battery) for i, start in enumerate(world.starts)]
    visited = np.zeros((n, n), dtype=bool)
    survivor_index = {cell: i for i, cell in enumerate(world.survivors)}
    found = [False] * len(world.survivors)
    for a in agents:
        visited[a.pos] = True
        if a.pos in survivor_index:
            found[survivor_index[a.pos]] = True

    reachable_mask = world.reachable_mask.copy()
    coverage = compute_coverage(visited, reachable_mask)
    shift_tick = max(1, cfg.shift_tick)
    comm_enabled = True
    collisions = wall_surprises = interlock_trips = 0
    tick_at_90: int | None = 0 if coverage >= COVERAGE_TARGET else None
    latencies: list[float] = []
    curve: list[float] = [coverage]
    frames: list[Frame] = [_snapshot(0, world, visited, agents, found, coverage)] if record_frames else []
    ticks_run = 0

    for tick in range(1, cfg.max_ticks + 1):
        ticks_run = tick
        if round_no == 2 and tick == shift_tick:
            apply_shift(world, cfg.new_walls, [a.pos for a in agents], shift_rng)
            if FAILED_ROBOT_ID < len(agents):
                agents[FAILED_ROBOT_ID].fail()
            comm_enabled = False
            reachable_mask = post_shift_reachable(world.grid, agents, visited)

        start = time.perf_counter()
        for a in agents:
            wall_surprises += a.sense(world.grid, cfg.sense_range)
        if comm_enabled:
            share_maps(agents, cfg.comm_range)
        radius = perception_radius(comm_enabled, cfg.comm_range)
        previous = [a.pos for a in agents]
        occupied = set(previous)
        for a in agents:
            blocked, others = perceive(a, agents, radius)
            move = a.choose_move(cfg, blocked, others, agent_rng)
            if move is not None and (world.grid[move] != FREE or move in occupied):
                interlock_trips += 1  # hardware safety interlock (should never fire)
                a.mode = MODE_BLOCKED
                move = None
            if move is not None:
                occupied.discard(a.pos)
                occupied.add(move)
            a.commit(move)
        latencies.append((time.perf_counter() - start) * 1000.0)

        collisions += count_collisions(agents, world.grid, previous)
        for a in agents:
            visited[a.pos] = True
            if a.pos in survivor_index:
                found[survivor_index[a.pos]] = True
        coverage = compute_coverage(visited, reachable_mask)
        curve.append(coverage)
        if tick_at_90 is None and coverage >= COVERAGE_TARGET:
            tick_at_90 = tick
        if record_frames:
            frames.append(_snapshot(tick, world, visited, agents, found, coverage))

        shift_pending = round_no == 2 and tick < shift_tick
        swarm_done = all(not a.active or a.mode == MODE_IDLE for a in agents)
        if not shift_pending and (swarm_done or (coverage >= 1.0 and all(found))):
            break

    energy = sum(a.moves for a in agents)
    ratio = sum(found) / len(found) if found else 1.0
    fitness = compute_fitness(coverage, ratio, tick_at_90, cfg.max_ticks, energy, collisions)
    return SimulationResult(
        round_no=round_no,
        seed=cfg.seed,
        coverage=coverage,
        survivors_found=sum(found),
        survivors_total=len(found),
        tick_at_90=tick_at_90,
        ticks_run=ticks_run,
        energy_moves=energy,
        collisions=collisions,
        fitness=fitness,
        max_latency_ms=max(latencies) if latencies else 0.0,
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        reachable_cells=int(reachable_mask.sum()),
        visited_cells=int((visited & reachable_mask).sum()),
        wall_surprises=wall_surprises,
        interlock_trips=interlock_trips,
        depleted_robots=sum(1 for a in agents if a.alive and a.battery <= 0),
        final_grid=world.grid.copy(),
        visited_mask=visited.copy(),
        final_positions=tuple(a.pos for a in agents),
        alive=tuple(a.alive for a in agents),
        survivors=tuple(world.survivors),
        found=tuple(found),
        coverage_curve=tuple(curve),
        frames=tuple(frames),
    )


def evaluate(cfg: SwarmConfig, seeds: list[int] | tuple[int, ...], round_no: int = 1) -> list[SimulationResult]:
    """Run the same configuration over several seeds."""
    return [simulate(cfg.with_updates(seed=s), round_no) for s in seeds]


def mean_fitness(cfg: SwarmConfig, seeds: list[int] | tuple[int, ...], round_no: int = 1) -> float:
    """Average fitness of ``cfg`` over ``seeds``."""
    return float(np.mean([r.fitness for r in evaluate(cfg, seeds, round_no)]))
