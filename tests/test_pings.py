"""Tests for survivor acoustic pings and the time-to-first-survivor metric."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from swarmrescue.agent import MODE_PHEROMONE, MODE_PING, Robot
from swarmrescue.config import SwarmConfig
from swarmrescue.simulation import MissionControl, simulate
from swarmrescue.world import DEBRIS, FREE


@pytest.fixture
def robot() -> Robot:
    """A robot at (2, 0) that has searched every cell around it."""
    r = Robot(0, (2, 0), 6, battery=30)
    r.known[:] = FREE
    r.trail.intensity[:] = 1.0
    r.trail.visited[:] = True
    return r


def test_ping_takes_priority_over_pheromone(robot: Robot) -> None:
    """A heard ping overrides the pheromone rule and heads toward the survivor."""
    robot.trail.visited[1, 0] = False  # an unsearched neighbour would attract the pheromone rule
    cfg = SwarmConfig(randomness=0.0)
    assert robot.choose_move(cfg, set(), [], np.random.default_rng(0)) == (1, 0)
    assert robot.mode == MODE_PHEROMONE
    move = robot.choose_move(cfg, set(), [], np.random.default_rng(0), pings=[(2, 5)])
    assert move == (2, 1) and robot.mode == MODE_PING


def test_ping_route_is_shortest_and_avoids_debris(robot: Robot) -> None:
    """Homing follows a shortest route around known debris."""
    robot.known[2, 1] = DEBRIS
    move = robot.home_in_on_ping([(2, 3)], set())
    assert move in {(1, 0), (3, 0)}
    assert robot.home_in_on_ping([], set()) is None


def test_ping_plans_through_unknown_space() -> None:
    """Sound travels through rubble: the robot plans optimistically through unknown cells."""
    r = Robot(0, (0, 0), 5, battery=10)
    r.sense_debris(np.zeros((5, 5), dtype=np.int8), 1)  # only the 2x2 corner is known
    assert r.home_in_on_ping([(4, 4)], set()) in {(0, 1), (1, 0)}


def test_ping_never_steps_into_blocked_cell(robot: Robot) -> None:
    """Homing stays collision-free: a teammate on the route forces a detour."""
    move = robot.home_in_on_ping([(2, 5)], {(2, 1)})
    assert move in {(1, 0), (3, 0)}


def test_heard_pings_respect_range_and_found_status() -> None:
    """Only unfound survivors within ping_range are heard, and only when enabled."""
    cfg = SwarmConfig(seed=3, ping_range=4)
    mission = MissionControl(cfg)
    r = mission.robots[1]
    heard = mission.heard_pings(r)
    for cell in heard:
        assert abs(cell[0] - r.pos[0]) + abs(cell[1] - r.pos[1]) <= 4
    target = mission.zone.survivors[0]
    r.pos = target.cell
    assert target.cell in mission.heard_pings(r)
    mission.found[target.survivor_id] = True
    assert target.cell not in mission.heard_pings(r)
    silent = MissionControl(cfg.with_updates(use_pings=False))
    silent.robots[1].pos = silent.zone.survivors[0].cell
    assert silent.heard_pings(silent.robots[1]) == []


def test_time_to_first_survivor_metric() -> None:
    """first_survivor_tick is the minimum of the per-survivor found ticks."""
    r = simulate(SwarmConfig(seed=1))
    ticks = [t for t in r.survivor_found_ticks if t is not None]
    assert len(ticks) == r.survivors_found
    assert r.first_survivor_tick == min(ticks)
    assert 0 <= r.first_survivor_tick <= r.ticks_run
    assert r.summary()["first_survivor_tick"] == r.first_survivor_tick


def test_pings_speed_up_rescue_on_average() -> None:
    """Averaged over 10 zones, pings find all survivors sooner (and never add collisions)."""

    def mean_all_found(use_pings: bool) -> float:
        total = 0.0
        for seed in range(1, 11):
            res = simulate(SwarmConfig(seed=seed, use_pings=use_pings))
            assert res.collisions == 0
            total += max(t if t is not None else res.ticks_run for t in res.survivor_found_ticks)
        return total / 10

    assert mean_all_found(True) < mean_all_found(False)


@pytest.mark.parametrize("bad", [{"ping_range": 0}, {"ping_range": 11}, {"use_pings": "yes"}])
def test_ping_config_validation(bad: dict[str, Any]) -> None:
    """Bad ping settings are rejected."""
    with pytest.raises(ValueError):
        SwarmConfig(**bad)
