"""Tests for configuration defaults and validation."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from swarmrescue.config import SwarmConfig


def test_defaults_match_specification(default_cfg: SwarmConfig) -> None:
    """Defaults equal the values given in the mission brief."""
    assert (default_cfg.grid_size, default_cfg.wall_density) == (20, 0.18)
    assert (default_cfg.num_agents, default_cfg.num_survivors) == (4, 5)
    assert (default_cfg.max_ticks, default_cfg.battery) == (300, 250)
    assert (default_cfg.comm_range, default_cfg.sense_range) == (5, 1)
    assert (default_cfg.pheromone_weight, default_cfg.spread_weight, default_cfg.randomness) == (1.0, 0.5, 0.1)
    assert (default_cfg.shift_tick, default_cfg.new_walls) == (60, 25)


def test_config_is_frozen(default_cfg: SwarmConfig) -> None:
    """Configuration objects are immutable."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        default_cfg.num_agents = 10  # type: ignore[misc]


def test_with_updates_returns_validated_copy(default_cfg: SwarmConfig) -> None:
    """with_updates copies and re-validates."""
    new = default_cfg.with_updates(num_agents=6)
    assert new.num_agents == 6 and default_cfg.num_agents == 4
    with pytest.raises(ValueError):
        default_cfg.with_updates(num_agents=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"grid_size": 2},
        {"wall_density": -0.1},
        {"wall_density": 0.9},
        {"num_agents": 0},
        {"num_survivors": -1},
        {"max_ticks": 0},
        {"battery": 0},
        {"comm_range": 0},
        {"sense_range": 0},
        {"pheromone_weight": 3.5},
        {"spread_weight": -0.1},
        {"randomness": 1.5},
        {"new_walls": -3},
        {"seed": -1},
        {"grid_size": 5.5},
        {"num_agents": True},
        {"pheromone_weight": "high"},
        {"grid_size": 6, "num_agents": 20, "num_survivors": 20},
    ],
)
def test_validation_rejects_bad_values(changes: dict[str, Any]) -> None:
    """Every out-of-range or wrongly typed value raises ValueError."""
    with pytest.raises(ValueError):
        SwarmConfig(**changes)
