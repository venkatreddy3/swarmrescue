"""Mission Control: the tick-based search-and-rescue simulator and fitness.

In one tick, every robot senses nearby debris, the radio link shares
pheromone trails between robots in range, and then the robots move one after
another. Sequential moves plus a per-robot occupancy check make the swarm
collision-free by construction. Mission Control still audits every tick and
counts any collision it finds. It is only an observer: robots never receive
orders from it, so it is not a central controller.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarmrescue.agent import MODE_BLOCKED, MODE_IDLE, Robot
from swarmrescue.config import SwarmConfig
from swarmrescue.coordination import RadioLink, perceive_teammates
from swarmrescue.learning import AdaptiveWeightLearner, weight_presets
from swarmrescue.telemetry import NO_NEIGHBOUR, MissionTelemetry
from swarmrescue.world import (
    DEBRIS,
    FREE,
    Aftershock,
    Cell,
    DisasterZone,
    drop_aftershock_debris,
    generate_disaster_zone,
    manhattan,
    reachable,
)

COVERAGE_TARGET: float = 0.9
FAILED_ROBOT_ID: int = 0


@dataclass(frozen=True)
class Frame:
    """Snapshot of the mission after one tick (used for animation).

    Attributes:
        tick: Tick number (0 = initial state).
        grid: Ground-truth occupancy grid at this tick.
        visited: Cells searched by any robot so far.
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
    """Outcome and metrics of one simulated search-and-rescue mission."""

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
    aftershock: Aftershock | None = None
    survivor_found_ticks: tuple[int | None, ...] = ()
    ping_detections: int = 0
    end_reason: str = ""
    min_separation: int = NO_NEIGHBOUR
    margin_violations: int = 0
    deadlocks_broken: int = 0
    blocked_ticks: int = 0
    latency_budget_violations: int = 0
    max_reroute_ms: float = 0.0
    reroute_events: int = 0
    mean_path_length: float = 0.0
    ticks_budget: int = 300

    @property
    def coverage_speed(self) -> float:
        """Coverage speed: 1 - tick_at_90 / ticks budget used by the fitness (0 if 90% never reached)."""
        if self.tick_at_90 is None:
            return 0.0
        return max(0.0, 1.0 - self.tick_at_90 / max(1, self.ticks_budget))

    @property
    def throughput(self) -> float:
        """Search throughput: reachable cells searched per tick."""
        return self.visited_cells / max(1, self.ticks_run)

    @property
    def coverage_velocity(self) -> float:
        """Coverage velocity: percentage points of the reachable area searched per tick."""
        return 100.0 * self.coverage / max(1, self.ticks_run)

    @property
    def end_message(self) -> str:
        """Operator-facing sentence explaining why and when the mission ended."""
        return format_end_message(
            self.ticks_run, self.end_reason, self.coverage, self.survivors_found, self.survivors_total, self.collisions
        )

    @property
    def first_survivor_tick(self) -> int | None:
        """Time-to-first-survivor: tick when the first survivor was found."""
        ticks = [t for t in self.survivor_found_ticks if t is not None]
        return min(ticks) if ticks else None

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
            "first_survivor_tick": self.first_survivor_tick,
            "throughput": round(self.throughput, 3),
            "coverage_velocity": round(self.coverage_velocity, 3),
            "min_separation": self.min_separation,
            "deadlocks_broken": self.deadlocks_broken,
            "max_reroute_ms": round(self.max_reroute_ms, 3),
        }


MissionReport = SimulationResult


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
    where ``speed = 1 - tick_at_90 / max_ticks`` is the coverage speed term (0 if 90% coverage was
    never reached), energy moves are the path length of all robots (energy efficiency term), and
    collisions must stay zero (collision-free trajectories).
    """
    speed = 0.0 if tick_at_90 is None else 1.0 - tick_at_90 / max_ticks
    return 100.0 * coverage + 20.0 * survivors_ratio + 20.0 * speed - 0.01 * energy_moves - 100.0 * collisions


REASON_COMPLETE = "every reachable cell searched and every survivor found"


def mission_end_reason(robots: list[Robot], complete: bool, time_up: bool, max_ticks: int) -> str:
    """Plain-English reason the mission stopped.

    Args:
        robots: The swarm at the end of the mission.
        complete: Full coverage reached and every survivor found.
        time_up: The time limit was reached.
        max_ticks: The mission time limit.

    Returns:
        A short reason such as ``"robot batteries depleted"``.
    """
    if complete:
        return REASON_COMPLETE
    if time_up:
        return f"time limit of {max_ticks} ticks reached"
    working = [r for r in robots if r.alive]
    if not working:
        return "all robots have failed"
    depleted = sum(1 for r in working if r.battery <= 0)
    if depleted == len(working):
        return "robot batteries depleted"
    if depleted:
        return (
            f"{depleted} of {len(working)} working robots out of battery; "
            "the rest have nothing left to search in their maps"
        )
    return "robots have nothing left to search in their maps (remaining cells are cut off or unknown)"


def format_end_message(tick: int, reason: str, coverage: float, found: int, total: int, collisions: int) -> str:
    """``"Mission ended at tick 253: robot batteries depleted (85% coverage, 4 of 5 survivors found, 0 collisions)."``

    Coverage is rounded *down*, so an unfinished search never reads as 100%.
    """
    verb = "complete" if reason == REASON_COMPLETE else "ended"
    pct = math.floor(coverage * 100 + 1e-9)
    return (
        f"Mission {verb} at tick {tick}: {reason} "
        f"({pct}% coverage, {found} of {total} survivors found, {collisions} collisions)."
    )


def compute_coverage(visited: np.ndarray, reachable_mask: np.ndarray) -> float:
    """Fraction of reachable free cells that have been searched, in [0, 1]."""
    total = int(reachable_mask.sum())
    if total == 0:
        return 1.0
    return int((visited & reachable_mask).sum()) / total


def count_collisions(robots: list[Robot], grid: np.ndarray, previous: list[Cell]) -> int:
    """Collision audit for one tick: shared cells, robots inside debris, head-on swaps."""
    positions = [r.pos for r in robots]
    collisions = len(positions) - len(set(positions))
    collisions += sum(1 for p in positions if grid[p] == DEBRIS)
    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            if positions[i] == previous[j] and positions[j] == previous[i] and positions[i] != previous[i]:
                collisions += 1
    return collisions


def post_aftershock_reachable(grid: np.ndarray, robots: list[Robot], visited: np.ndarray) -> np.ndarray:
    """Recompute the coverage denominator after the aftershock.

    Reachable means free cells connected to a working robot (failed robots
    count as obstacles), plus already-searched cells that are still free,
    because that search work was genuinely done.
    """
    blocked = grid.copy()
    for r in robots:
        if not r.alive:
            blocked[r.pos] = DEBRIS
    mask = reachable(blocked, [r.pos for r in robots if r.alive])
    result: np.ndarray = mask | (visited & (grid == FREE))
    return result


class MissionControl:
    """Runs one search-and-rescue mission and records its metrics.

    Mission Control owns the ground truth (the :class:`DisasterZone`), the
    clock, the Round 2 :class:`Aftershock` and the safety audit. The robots
    decide every move themselves.
    """

    def __init__(self, cfg: SwarmConfig, round_no: int = 1, record_frames: bool = False) -> None:
        """Prepare a mission.

        Args:
            cfg: Mission configuration (``cfg.seed`` drives all randomness).
            round_no: 1 = static building; 2 = aftershock at ``cfg.shift_tick``
                (new debris, robot 0 fails, radio link lost).
            record_frames: Store a :class:`Frame` per tick for animation.

        Raises:
            ValueError: If ``round_no`` is not 1 or 2.
        """
        if round_no not in (1, 2):
            raise ValueError("round_no must be 1 or 2")
        self.cfg = cfg
        self.round_no = round_no
        self.record_frames = record_frames
        zone_ss, robot_ss, shock_ss = np.random.SeedSequence(cfg.seed).spawn(3)
        self.zone: DisasterZone = generate_disaster_zone(cfg, np.random.default_rng(zone_ss))
        self.robot_rng = np.random.default_rng(robot_ss)
        self.aftershock_rng = np.random.default_rng(shock_ss)
        n = cfg.grid_size
        self.robots = [Robot(i, cell, n, cfg.battery) for i, cell in enumerate(self.zone.entry_cells)]
        if cfg.use_learning:
            self._attach_learners()
        self.radio = RadioLink(cfg.comm_range)
        self.visited = np.zeros((n, n), dtype=bool)
        self.survivor_index = {s.cell: s.survivor_id for s in self.zone.survivors}
        self.found = [False] * len(self.zone.survivors)
        self.found_ticks: list[int | None] = [None] * len(self.zone.survivors)
        self.ping_detections = 0
        self.tick = 0
        self.aftershock: Aftershock | None = None
        self.aftershock_tick = max(1, cfg.shift_tick)
        self.reachable_mask = self.zone.reachable_mask.copy()
        self.collisions = 0
        self.wall_surprises = 0
        self.interlock_trips = 0
        self.telemetry = MissionTelemetry(cfg.latency_budget_ms, cfg.safety_margin)
        self._record_search()

    def _attach_learners(self) -> None:
        """Decentralized online learning: one independent, seeded bandit per robot (no central learner)."""
        presets = weight_presets(self.cfg.pheromone_weight, self.cfg.spread_weight)
        for r in self.robots:
            rng = np.random.default_rng([self.cfg.seed, r.robot_id, 7919])
            r.learner = AdaptiveWeightLearner(presets, self.cfg.learning_epsilon, self.cfg.learning_window, rng)

    @property
    def latencies(self) -> list[float]:
        """Per-tick decision latency in milliseconds."""
        return self.telemetry.latencies

    def _record_search(self) -> None:
        """Mark robot cells as searched and rescue any survivor found there."""
        for r in self.robots:
            self.visited[r.pos] = True
            sid = self.survivor_index.get(r.pos)
            if sid is not None and not self.found[sid]:
                self.found[sid] = True
                self.found_ticks[sid] = self.tick

    def heard_pings(self, robot: Robot) -> list[Cell]:
        """Acoustic pings ``robot`` hears: unfound survivors within ``ping_range``.

        Survivors stop pinging once found (they have been reached).
        """
        if not self.cfg.use_pings or not robot.active:
            return []
        return [
            s.cell
            for s in self.zone.survivors
            if not self.found[s.survivor_id] and manhattan(robot.pos, s.cell) <= self.cfg.ping_range
        ]

    def coverage(self) -> float:
        """Current coverage of the reachable disaster zone."""
        return compute_coverage(self.visited, self.reachable_mask)

    def trigger_aftershock(self, tick: int) -> Aftershock:
        """Round 2: drop debris, knock out robot 0 and cut the radio link.

        This stress-tests the swarm for a single point of failure. The robots
        are not told what changed; they must sense the debris and reroute.
        """
        debris = drop_aftershock_debris(
            self.zone, self.cfg.new_walls, [r.pos for r in self.robots], self.aftershock_rng
        )
        failed = None
        if len(self.robots) > FAILED_ROBOT_ID:
            self.robots[FAILED_ROBOT_ID].fail()
            failed = FAILED_ROBOT_ID
        self.radio.cut()
        self.reachable_mask = post_aftershock_reachable(self.zone.grid, self.robots, self.visited)
        self.aftershock = Aftershock(tick, tuple(debris), failed, radio_lost=True)
        return self.aftershock

    def step(self, tick: int) -> None:
        """Advance the mission by one tick."""
        self.tick = tick
        if self.round_no == 2 and tick == self.aftershock_tick:
            self.trigger_aftershock(tick)
        start = time.perf_counter()
        self._sense_and_share()
        previous = [r.pos for r in self.robots]
        self._move_robots(set(previous))
        self.telemetry.record_tick((time.perf_counter() - start) * 1000.0)
        self.telemetry.record_positions([r.pos for r in self.robots])
        self.collisions += count_collisions(self.robots, self.zone.grid, previous)
        self._record_search()

    def _sense_and_share(self) -> None:
        """Every robot senses debris, trails evaporate (optional), the radio link shares trails."""
        for r in self.robots:
            self.wall_surprises += r.sense_debris(self.zone.grid, self.cfg.sense_range)
            if self.cfg.use_evaporation and r.alive:
                r.trail.evaporate(self.cfg.evaporation_rate)
        self.radio.share_pheromone_trails(self.robots)

    def _move_robots(self, occupied: set[Cell]) -> None:
        """Robots decide and move one after another (collision-free by construction)."""
        radius = self.radio.perception_radius
        positions = np.asarray([r.pos for r in self.robots])
        for i, r in enumerate(self.robots):
            blocked, others = perceive_teammates(r, self.robots, radius, positions)
            pings = self.heard_pings(r)
            self.ping_detections += bool(pings)
            began = time.perf_counter()
            move = r.choose_move(self.cfg, blocked, others, self.robot_rng, pings)
            self.telemetry.record_decision(r.mode, (time.perf_counter() - began) * 1000.0, r.last_surprises > 0)
            if move is not None and (self.zone.grid[move] != FREE or move in occupied):
                self.interlock_trips += 1  # hardware safety interlock (should never fire)
                r.mode = MODE_BLOCKED
                move = None
            if move is not None:
                occupied.discard(r.pos)
                occupied.add(move)
            r.commit(move)
            positions[i] = r.pos

    def snapshot(self, tick: int, coverage: float) -> Frame:
        """Build an immutable :class:`Frame` of the current state."""
        return Frame(
            tick=tick,
            grid=self.zone.grid.copy(),
            visited=self.visited.copy(),
            positions=tuple(r.pos for r in self.robots),
            alive=tuple(r.alive for r in self.robots),
            found=tuple(self.found),
            coverage=coverage,
        )

    def swarm_done(self) -> bool:
        """True when no robot can or wants to move (failed, flat battery or idle)."""
        return all(not r.active or r.mode == MODE_IDLE for r in self.robots)

    def run(self) -> SimulationResult:
        """Run the mission to completion and return its metrics."""
        start = self.coverage()
        progress = _Progress(start, [self.snapshot(0, start)] if self.record_frames else [])
        for tick in range(1, self.cfg.max_ticks + 1):
            self.step(tick)
            progress.update(tick, self.coverage(), self.snapshot if self.record_frames else None)
            progress.complete = progress.coverage >= 1.0 and all(self.found)
            aftershock_pending = self.round_no == 2 and tick < self.aftershock_tick
            if not aftershock_pending and (self.swarm_done() or progress.complete):
                break
        return self._result(progress)

    def _result(self, p: _Progress) -> SimulationResult:
        """Package the finished mission as a :class:`SimulationResult`."""
        cfg = self.cfg
        time_up = not p.complete and p.ticks_run >= cfg.max_ticks and not self.swarm_done()
        energy = sum(r.moves for r in self.robots)
        ratio = sum(self.found) / len(self.found) if self.found else 1.0
        return SimulationResult(
            round_no=self.round_no,
            seed=cfg.seed,
            coverage=p.coverage,
            survivors_found=sum(self.found),
            survivors_total=len(self.found),
            tick_at_90=p.tick_at_90,
            ticks_run=p.ticks_run,
            energy_moves=energy,
            collisions=self.collisions,
            fitness=compute_fitness(p.coverage, ratio, p.tick_at_90, cfg.max_ticks, energy, self.collisions),
            max_latency_ms=max(self.latencies, default=0.0),
            mean_latency_ms=float(np.mean(self.latencies)) if self.latencies else 0.0,
            reachable_cells=int(self.reachable_mask.sum()),
            visited_cells=int((self.visited & self.reachable_mask).sum()),
            wall_surprises=self.wall_surprises,
            interlock_trips=self.interlock_trips,
            depleted_robots=sum(1 for r in self.robots if r.alive and r.battery <= 0),
            final_grid=self.zone.grid.copy(),
            visited_mask=self.visited.copy(),
            final_positions=tuple(r.pos for r in self.robots),
            alive=tuple(r.alive for r in self.robots),
            survivors=tuple(self.zone.survivor_cells),
            found=tuple(self.found),
            coverage_curve=tuple(p.curve),
            frames=tuple(p.frames),
            aftershock=self.aftershock,
            survivor_found_ticks=tuple(self.found_ticks),
            ping_detections=self.ping_detections,
            end_reason=mission_end_reason(self.robots, p.complete, time_up, cfg.max_ticks),
            ticks_budget=cfg.max_ticks,
            **self._safety_metrics(energy),
        )

    def _safety_metrics(self, energy: int) -> dict[str, Any]:
        """Hard-constraint and objective metrics gathered by the telemetry recorder."""
        t = self.telemetry
        return {
            "min_separation": t.min_separation,
            "margin_violations": t.margin_violations,
            "deadlocks_broken": t.deadlocks_broken,
            "blocked_ticks": t.blocked_ticks,
            "latency_budget_violations": t.budget_violations,
            "max_reroute_ms": t.max_reroute_ms,
            "reroute_events": len(t.reroute_ms),
            "mean_path_length": energy / max(1, len(self.robots)),
        }


@dataclass
class _Progress:
    """Per-tick bookkeeping of a running mission (coverage curve, milestones, frames)."""

    coverage: float
    frames: list[Frame]
    ticks_run: int = 0
    complete: bool = False
    curve: list[float] = field(default_factory=list)
    tick_at_90: int | None = None

    def __post_init__(self) -> None:
        """Start the curve at tick 0."""
        self.curve.append(self.coverage)
        if self.coverage >= COVERAGE_TARGET:
            self.tick_at_90 = 0

    def update(self, tick: int, coverage: float, snapshot: Callable[[int, float], Frame] | None) -> None:
        """Record the state after ``tick``."""
        self.ticks_run, self.coverage = tick, coverage
        self.curve.append(coverage)
        if self.tick_at_90 is None and coverage >= COVERAGE_TARGET:
            self.tick_at_90 = tick
        if snapshot is not None:
            self.frames.append(snapshot(tick, coverage))


def simulate(cfg: SwarmConfig, round_no: int = 1, record_frames: bool = False) -> SimulationResult:
    """Run one full search-and-rescue mission (a wrapper around :class:`MissionControl`).

    Raises:
        ValueError: If ``round_no`` is not 1 or 2.
    """
    return MissionControl(cfg, round_no, record_frames).run()


def evaluate(cfg: SwarmConfig, seeds: list[int] | tuple[int, ...], round_no: int = 1) -> list[SimulationResult]:
    """Run the same configuration over several seeds (disaster zones)."""
    return [simulate(cfg.with_updates(seed=s), round_no) for s in seeds]


def mean_fitness(cfg: SwarmConfig, seeds: list[int] | tuple[int, ...], round_no: int = 1) -> float:
    """Average fitness of ``cfg`` over ``seeds``."""
    return float(np.mean([r.fitness for r in evaluate(cfg, seeds, round_no)]))
