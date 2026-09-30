"""Tests for the mission simulator and fitness function."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from swarmrescue.config import SwarmConfig
from swarmrescue.simulation import (
    MissionControl,
    SimulationResult,
    compute_coverage,
    compute_fitness,
    simulate,
)
from swarmrescue.world import FREE, WALL, Aftershock

MANY_SEEDS = range(25)


@pytest.fixture(scope="module")
def round1() -> SimulationResult:
    """Default Round 1 mission with frames recorded."""
    return simulate(SwarmConfig(seed=1), round_no=1, record_frames=True)


@pytest.fixture(scope="module")
def round2() -> SimulationResult:
    """Default Round 2 mission with frames recorded."""
    return simulate(SwarmConfig(seed=1), round_no=2, record_frames=True)


def test_fitness_formula_exact() -> None:
    """Fitness equals the published formula term by term."""
    value = compute_fitness(0.8, 0.6, 150, 300, 500, 0)
    assert value == pytest.approx(100 * 0.8 + 20 * 0.6 + 20 * (1 - 150 / 300) - 0.01 * 500)
    assert value == pytest.approx(97.0)
    assert compute_fitness(1.0, 1.0, None, 300, 0, 0) == pytest.approx(120.0)
    assert compute_fitness(1.0, 1.0, 0, 300, 0, 2) == pytest.approx(140.0 - 200.0)


@pytest.mark.parametrize("round_no", [1, 2])
def test_result_fitness_is_consistent(round_no: int) -> None:
    """The reported fitness is recomputable from the reported metrics."""
    cfg = SwarmConfig(seed=3)
    r = simulate(cfg, round_no)
    assert r.fitness == pytest.approx(
        compute_fitness(r.coverage, r.survivors_ratio, r.tick_at_90, cfg.max_ticks, r.energy_moves, r.collisions)
    )
    assert r.coverage == pytest.approx(r.visited_cells / r.reachable_cells)


def test_coverage_helper_bounds() -> None:
    """compute_coverage is a ratio in [0, 1]."""
    reach = np.ones((4, 4), dtype=bool)
    visited = np.zeros((4, 4), dtype=bool)
    assert compute_coverage(visited, reach) == 0.0
    visited[:2] = True
    assert compute_coverage(visited, reach) == 0.5
    visited[:] = True
    assert compute_coverage(visited, reach) == 1.0


@pytest.mark.parametrize("round_no", [1, 2])
def test_zero_collisions_and_valid_coverage_many_seeds(small_cfg: SwarmConfig, round_no: int) -> None:
    """Across many seeds: no collisions, no interlock trips, coverage in [0, 1]."""
    for seed in MANY_SEEDS:
        r = simulate(small_cfg.with_updates(seed=seed), round_no)
        assert r.collisions == 0, f"seed {seed}"
        assert r.interlock_trips == 0, f"seed {seed}"
        assert 0.0 <= r.coverage <= 1.0
        assert all(0.0 <= c <= 1.0 for c in r.coverage_curve)
        assert 0 <= r.survivors_found <= r.survivors_total


@pytest.mark.parametrize("round_no", [1, 2])
def test_zero_collisions_default_config(round_no: int) -> None:
    """Default-size missions are collision-free too."""
    for seed in range(1, 9):
        assert simulate(SwarmConfig(seed=seed), round_no).collisions == 0


@pytest.mark.parametrize("fixture_name", ["round1", "round2"])
def test_robots_never_occupy_walls_or_share_cells(fixture_name: str, request: pytest.FixtureRequest) -> None:
    """In every recorded frame, robots stand on distinct free cells and move <= 1 step."""
    r: SimulationResult = request.getfixturevalue(fixture_name)
    for prev, frame in pairwise(r.frames):
        assert len(set(frame.positions)) == len(frame.positions)
        for old, new in zip(prev.positions, frame.positions, strict=True):
            assert frame.grid[new] == FREE
            assert abs(old[0] - new[0]) + abs(old[1] - new[1]) <= 1


def test_round1_explores_default_map(round1: SimulationResult) -> None:
    """Round 1 with default settings reaches full coverage and finds everyone."""
    assert round1.coverage >= 0.95
    assert round1.survivors_found == round1.survivors_total
    assert round1.tick_at_90 is not None


def test_round2_shift_effects(round2: SimulationResult) -> None:
    """At the shift: 25 new walls, robot 0 fails and never moves again."""
    cfg = SwarmConfig()
    before = round2.frames[cfg.shift_tick - 1]
    after = round2.frames[cfg.shift_tick]
    new_walls = (after.grid == WALL) & (before.grid == FREE)
    assert int(new_walls.sum()) == cfg.new_walls
    assert before.alive[0] and not after.alive[0]
    frozen = {f.positions[0] for f in round2.frames[cfg.shift_tick - 1 :]}
    assert len(frozen) == 1
    assert all(f.alive[1:] == (True,) * (cfg.num_agents - 1) for f in round2.frames)
    assert round2.wall_surprises > 0  # robots discovered debris on known routes
    assert round2.reachable_cells <= int((round2.final_grid == FREE).sum())


def test_replanning_after_shift_keeps_exploring(round2: SimulationResult) -> None:
    """Robots keep covering new cells after debris falls (they replan)."""
    cfg = SwarmConfig()
    at_shift = round2.frames[cfg.shift_tick].visited.sum()
    assert round2.frames[-1].visited.sum() > at_shift
    assert round2.coverage >= 0.85


def test_latency_per_tick_under_50ms(round1: SimulationResult, round2: SimulationResult) -> None:
    """Decentralized decisions for the whole swarm stay well under 50 ms per tick."""
    assert round1.max_latency_ms < 50.0
    assert round2.max_latency_ms < 50.0


def test_battery_limits_energy() -> None:
    """Energy never exceeds total battery; a tiny battery stops robots early."""
    cfg = SwarmConfig(seed=2, battery=10)
    r = simulate(cfg)
    assert r.energy_moves <= cfg.battery * cfg.num_agents
    assert r.depleted_robots >= 1


@pytest.mark.parametrize("round_no", [1, 2])
def test_deterministic_with_same_seed(round_no: int) -> None:
    """Identical config and seed reproduce identical missions."""
    a = simulate(SwarmConfig(seed=9), round_no)
    b = simulate(SwarmConfig(seed=9), round_no)
    timing = {"max_latency_ms", "max_reroute_ms"}  # wall-clock measurements naturally vary
    assert {k: v for k, v in a.summary().items() if k not in timing} == {
        k: v for k, v in b.summary().items() if k not in timing
    }
    assert np.array_equal(a.visited_mask, b.visited_mask)
    assert a.final_positions == b.final_positions
    c = simulate(SwarmConfig(seed=10), round_no)
    assert not np.array_equal(a.final_grid, c.final_grid)


def test_invalid_round_rejected() -> None:
    """Only rounds 1 and 2 exist."""
    with pytest.raises(ValueError):
        simulate(SwarmConfig(), round_no=3)


def test_mission_control_records_aftershock(default_cfg: SwarmConfig) -> None:
    """Round 2 records one Aftershock: debris count, failed robot, radio lost."""
    mission = MissionControl(default_cfg.with_updates(seed=4), round_no=2)
    result = mission.run()
    shock = result.aftershock
    assert isinstance(shock, Aftershock)
    assert shock.tick == default_cfg.shift_tick
    assert len(shock.debris) == default_cfg.new_walls
    assert shock.failed_robot_id == 0 and shock.radio_lost
    assert not mission.radio.online and not mission.robots[0].alive
    assert simulate(default_cfg.with_updates(seed=4)).aftershock is None


def test_mission_control_matches_simulate(default_cfg: SwarmConfig) -> None:
    """simulate() is a thin wrapper around MissionControl.run()."""
    a = MissionControl(default_cfg.with_updates(seed=5)).run()
    b = simulate(default_cfg.with_updates(seed=5))
    assert a.fitness == b.fitness and a.final_positions == b.final_positions


def test_end_message_format_exact() -> None:
    """The operator message matches the agreed wording; coverage is rounded down."""
    from swarmrescue.simulation import REASON_COMPLETE, format_end_message

    assert format_end_message(253, "robot batteries depleted", 0.857, 4, 5, 0) == (
        "Mission ended at tick 253: robot batteries depleted (85% coverage, 4 of 5 survivors found, 0 collisions)."
    )
    assert "(99% coverage" in format_end_message(10, "x", 0.9999, 1, 1, 0)
    assert format_end_message(142, REASON_COMPLETE, 1.0, 5, 5, 0).startswith("Mission complete at tick 142:")


@pytest.mark.parametrize(
    ("changes", "round_no", "expected"),
    [
        ({"seed": 1}, 1, "every reachable cell searched and every survivor found"),
        ({"seed": 1, "battery": 60}, 1, "robot batteries depleted"),
        ({"seed": 1, "max_ticks": 80}, 1, "time limit of 80 ticks reached"),
        ({"seed": 1, "use_learning": False}, 2, "robot batteries depleted"),
    ],
)
def test_mission_end_reason_is_real(changes: dict[str, Any], round_no: int, expected: str) -> None:
    """The reported reason matches what actually stopped the mission."""
    r = simulate(SwarmConfig(**changes), round_no)
    assert r.end_reason == expected
    assert r.end_message.startswith(f"Mission {'complete' if 'every' in expected else 'ended'} at tick {r.ticks_run}: ")
    if expected == "robot batteries depleted":
        assert r.depleted_robots == sum(r.alive)


def test_end_reason_for_failed_and_mixed_swarms() -> None:
    """All-failed and partly-depleted swarms get their own reasons."""
    from swarmrescue.agent import Robot
    from swarmrescue.simulation import mission_end_reason

    a, b = Robot(0, (0, 0), 5, 10), Robot(1, (0, 1), 5, 0)
    assert mission_end_reason([a, b], False, False, 300).startswith("1 of 2 working robots out of battery")
    a.fail()
    b.fail()
    assert mission_end_reason([a, b], False, False, 300) == "all robots have failed"


def test_ablation_script_runs() -> None:
    """scripts/ablation.py produces one Markdown row per variant for both rounds."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "ablation.py"
    out = subprocess.run(
        [sys.executable, str(script), "--maps", "1"], capture_output=True, text=True, check=True
    ).stdout
    assert out.count("| Attempt 1 baseline") == 2 and out.count("| + acoustic pings only") == 2
