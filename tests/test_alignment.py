"""Tests for the Track 05 hard constraints and objectives made explicit in Attempt 2.

Safety margin, deadlock detection counters, latency-budget checks, sub-second
trajectory recalibration, throughput / coverage velocity and path length.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from swarmrescue.agent import MODE_BLOCKED, MODE_DEADLOCK, MODE_PHEROMONE, Robot
from swarmrescue.config import SwarmConfig
from swarmrescue.simulation import SimulationResult, simulate
from swarmrescue.telemetry import NO_NEIGHBOUR, MissionTelemetry, min_pairwise_distance
from swarmrescue.world import FREE

SEEDS = range(1, 6)


def adjacency_share(margin: int, round_no: int) -> float:
    """Fraction of ticks in which some pair of robots is adjacent (distance <= 1)."""
    close = total = 0
    for seed in SEEDS:
        r = simulate(SwarmConfig(safety_margin=margin, seed=seed), round_no, record_frames=True)
        assert r.collisions == 0
        close += sum(1 for f in r.frames[1:] if min_pairwise_distance(list(f.positions)) <= 1)
        total += len(r.frames) - 1
    return close / total


def test_min_pairwise_distance() -> None:
    """Exact minimum Manhattan distance; a lone robot has no neighbour."""
    assert min_pairwise_distance([(0, 0), (3, 4), (0, 2)]) == 2
    assert min_pairwise_distance([(1, 1)]) == NO_NEIGHBOUR


def test_clearance_rule() -> None:
    """A move may not bring the robot within the margin unless it keeps (or improves) clearance."""
    robot = Robot(0, (2, 2), 7, battery=10)
    robot.known[:] = FREE
    assert robot._keeps_clearance((2, 3), {(2, 5)}, margin=1)  # distance 2 > 1
    assert not robot._keeps_clearance((2, 3), {(2, 4)}, margin=1)  # would become adjacent (distance 1)
    assert robot._keeps_clearance((2, 1), {(2, 3)}, margin=1)  # moving away never blocked
    assert robot._keeps_clearance((2, 3), {(2, 4)}, margin=0)  # margin 0 = only no shared cell


def test_safety_margin_reduces_close_contact_without_collisions() -> None:
    """safety_margin=1 cuts the share of ticks with adjacent robots by more than half."""
    for round_no in (1, 2):
        assert adjacency_share(1, round_no) < 0.5 * adjacency_share(0, round_no)


def test_margin_blocks_planned_step_but_deadlock_breaker_may_relax_it() -> None:
    """A planned step into the margin becomes a wait; after 3 waits the breaker still moves the robot."""
    robot = Robot(0, (0, 0), 5, battery=20)
    robot.known[:] = FREE
    robot.trail.visited[:] = True
    robot.trail.visited[0, 4] = False
    cfg = SwarmConfig(randomness=0.0, safety_margin=2)
    blocked = {(1, 1)}  # a teammate diagonally adjacent: every neighbour is within the margin
    assert robot.choose_move(cfg, blocked, [], np.random.default_rng(0)) is None
    assert robot.mode == MODE_BLOCKED
    robot.stuck_ticks = 3
    move = robot.choose_move(cfg, blocked, [], np.random.default_rng(0))
    assert robot.mode == MODE_DEADLOCK and move in {(0, 1), (1, 0)}


def test_telemetry_counts_deadlocks_and_blocked_ticks() -> None:
    """Deadlock detection counters are explicit numbers, not just behaviour."""
    t = MissionTelemetry(latency_budget_ms=50.0, safety_margin=0)
    for mode in (MODE_DEADLOCK, MODE_BLOCKED, MODE_BLOCKED, MODE_PHEROMONE):
        t.record_decision(mode, 0.1, recalibrating=False)
    assert (t.deadlocks_broken, t.blocked_ticks) == (1, 2)


def test_latency_budget_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Exceeding latency_budget_ms is counted and logged once as a warning."""
    t = MissionTelemetry(latency_budget_ms=5.0, safety_margin=0)
    with caplog.at_level(logging.WARNING, logger="swarmrescue.telemetry"):
        for ms in (1.0, 7.5, 9.0):
            t.record_tick(ms)
    assert t.budget_violations == 2
    assert sum("over the 5 ms budget" in rec.getMessage() for rec in caplog.records) == 1


def test_default_missions_meet_the_real_time_budget() -> None:
    """No tick of the default missions exceeds the 50 ms budget."""
    for round_no in (1, 2):
        r = simulate(SwarmConfig(seed=2), round_no)
        assert r.latency_budget_violations == 0 and r.max_latency_ms < 50.0


def test_trajectory_recalibration_is_sub_second() -> None:
    """After the aftershock, robots that sense debris on known routes replan in well under a second."""
    r = simulate(SwarmConfig(seed=1), round_no=2)
    assert r.reroute_events > 0
    assert r.max_reroute_ms < 1000.0
    assert simulate(SwarmConfig(seed=1), round_no=1).reroute_events == 0  # no surprises without an aftershock


@pytest.fixture(scope="module")
def result() -> SimulationResult:
    """A default Round 1 mission."""
    return simulate(SwarmConfig(seed=4))


def test_throughput_and_coverage_velocity(result: SimulationResult) -> None:
    """Throughput = searched cells per tick; coverage velocity = coverage points per tick."""
    assert result.throughput == pytest.approx(result.visited_cells / result.ticks_run)
    assert result.coverage_velocity == pytest.approx(100 * result.coverage / result.ticks_run)
    assert result.summary()["throughput"] == round(result.throughput, 3)


def test_mean_path_length_is_energy_per_robot(result: SimulationResult) -> None:
    """Energy and path length are the same currency: one move per cell travelled."""
    assert result.mean_path_length == pytest.approx(result.energy_moves / 4)


@pytest.mark.parametrize("bad", [{"safety_margin": -1}, {"safety_margin": 4}, {"latency_budget_ms": 0.5}])
def test_new_parameters_are_validated(bad: dict[str, int | float]) -> None:
    """The new constraint parameters are bounded like every other field."""
    with pytest.raises(ValueError):
        SwarmConfig(**bad)  # type: ignore[arg-type]
