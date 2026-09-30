"""Tests for the pheromone-evaporation feature (trails fade over time)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from swarmrescue.agent import MODE_IDLE, MODE_PATROL, STALE_PHEROMONE, PheromoneTrail, Robot
from swarmrescue.config import SwarmConfig
from swarmrescue.optimizer import search_space
from swarmrescue.simulation import simulate
from swarmrescue.world import FREE


def test_evaporation_is_geometric_decay() -> None:
    """After k ticks at rate r, intensity is exactly (1 - r)^k times the deposit."""
    trail = PheromoneTrail(3)
    trail.deposit((1, 1), 2.0)
    for _ in range(10):
        trail.evaporate(0.1)
    assert trail.strength((1, 1)) == pytest.approx(2.0 * 0.9**10)
    assert trail.is_visited((1, 1))  # memory of the search is kept


def test_zero_rate_changes_nothing() -> None:
    """A rate of zero leaves the trail untouched."""
    trail = PheromoneTrail(3)
    trail.deposit((0, 0))
    trail.evaporate(0.0)
    assert trail.strength((0, 0)) == 1.0


def test_stale_mask_marks_faded_cells_only() -> None:
    """Only searched cells whose pheromone fell below the threshold are stale."""
    trail = PheromoneTrail(3)
    trail.deposit((0, 0), STALE_PHEROMONE / 2)
    trail.deposit((1, 1), 1.0)
    stale = trail.stale_mask()
    assert stale[0, 0] and not stale[1, 1] and not stale[2, 2]


def test_robot_patrols_to_stale_area_only_with_evaporation() -> None:
    """With evaporation on, an idle robot heads back to a long-faded cell."""
    robot = Robot(0, (2, 2), 5, battery=20)
    robot.known[:] = FREE
    robot.trail.intensity[:] = 1.0
    robot.trail.visited[:] = True
    robot.trail.intensity[2, 4] = 0.001  # searched long ago
    off = SwarmConfig(randomness=0.0, use_evaporation=False)
    assert robot.choose_move(off, set(), [], np.random.default_rng(0)) is None
    assert robot.mode == MODE_IDLE
    on = SwarmConfig(randomness=0.0, use_evaporation=True)
    assert robot.choose_move(on, set(), [], np.random.default_rng(0)) == (2, 3)
    assert robot.mode == MODE_PATROL


def test_evaporation_flag_changes_mission_but_stays_safe() -> None:
    """Evaporation is used by the simulator and keeps missions collision-free."""
    base = SwarmConfig(seed=2)
    on = simulate(base.with_updates(use_evaporation=True, evaporation_rate=0.05), round_no=2)
    off = simulate(base, round_no=2)
    assert on.collisions == 0 and 0.0 <= on.coverage <= 1.0
    assert (on.coverage, on.coverage_curve) != (off.coverage, off.coverage_curve)


def test_pso_search_space_includes_rate_only_when_enabled() -> None:
    """evaporation_rate joins the PSO search space only when the flag is on."""
    assert "evaporation_rate" not in search_space(SwarmConfig())
    assert search_space(SwarmConfig(use_evaporation=True))[-1] == "evaporation_rate"


@pytest.mark.parametrize("bad", [{"evaporation_rate": 0.5}, {"evaporation_rate": -0.01}, {"use_evaporation": 1}])
def test_evaporation_config_validation(bad: dict[str, Any]) -> None:
    """Out-of-range rates and non-boolean flags are rejected."""
    with pytest.raises(ValueError):
        SwarmConfig(**bad)
