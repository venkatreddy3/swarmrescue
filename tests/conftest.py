"""Shared pytest fixtures for the SwarmRescue test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmrescue.config import SwarmConfig


@pytest.fixture
def default_cfg() -> SwarmConfig:
    """The default mission configuration from the problem statement."""
    return SwarmConfig()


@pytest.fixture
def small_cfg() -> SwarmConfig:
    """A smaller, faster configuration for many-seed property tests."""
    return SwarmConfig(
        grid_size=12,
        num_agents=3,
        num_survivors=3,
        max_ticks=120,
        battery=120,
        shift_tick=30,
        new_walls=8,
    )
