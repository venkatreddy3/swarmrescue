"""Tests for the efficiency work: vectorised hot paths and the validated BFS route cache."""

from __future__ import annotations

import numpy as np
import pytest

from swarmrescue.agent import MODE_BFS, PheromoneTrail, Robot, crowding, crowding_many
from swarmrescue.config import SwarmConfig
from swarmrescue.coordination import RadioLink, perceive_teammates
from swarmrescue.simulation import MissionControl
from swarmrescue.world import DEBRIS, FREE

CFG = SwarmConfig(randomness=0.0)


@pytest.fixture
def corridor_robot() -> Robot:
    """A robot at (2, 0) in a known 7x7 room whose only unsearched cell is (2, 6)."""
    robot = Robot(0, (2, 0), 7, battery=50)
    robot.known[:] = FREE
    robot.trail.visited[:] = True
    robot.trail.intensity[:] = 1.0
    robot.trail.visited[2, 6] = False
    robot.trail.intensity[2, 6] = 0.0
    return robot


def step(robot: Robot, blocked: set[tuple[int, int]] | None = None) -> tuple[int, int] | None:
    """One decide-and-commit cycle with no teammates."""
    move = robot.choose_move(CFG, blocked or set(), [], np.random.default_rng(0))
    robot.commit(move)
    return move


def test_route_is_reused_while_belief_is_unchanged(corridor_robot: Robot) -> None:
    """BFS runs once; later steps come from the cache and still follow the shortest route."""
    moves = [step(corridor_robot) for _ in range(6)]
    assert moves == [(2, c) for c in range(1, 7)]  # the goal is reached in exactly the BFS distance (6)
    assert corridor_robot.bfs_calls == 1
    assert corridor_robot.route_cache_hits == 4  # steps 2-5; step 6 is the pheromone rule (goal adjacent)
    assert corridor_robot.mode != MODE_BFS


def test_new_debris_on_route_forces_a_new_bfs(corridor_robot: Robot) -> None:
    """Sensing debris changes the belief version, so the cached route is dropped."""
    step(corridor_robot)
    grid = np.zeros((7, 7), dtype=np.int8)
    grid[2, 2] = DEBRIS
    corridor_robot.sense_debris(grid, 1)
    move = step(corridor_robot)
    assert corridor_robot.bfs_calls == 2
    assert move in {(1, 1), (3, 1)}  # detour around the new debris


def test_teammate_on_route_forces_a_new_bfs(corridor_robot: Robot) -> None:
    """A perceived robot standing on the route invalidates it (collision-free rerouting)."""
    step(corridor_robot)
    move = step(corridor_robot, blocked={(2, 4)})
    assert corridor_robot.bfs_calls == 2 and move in {(1, 1), (3, 1)}


def test_goal_searched_by_teammate_invalidates_route(corridor_robot: Robot) -> None:
    """A merged trail that marks the goal as searched changes the belief and the route."""
    step(corridor_robot)
    teammate = PheromoneTrail(7)
    teammate.deposit((2, 6))
    version = corridor_robot.belief_version
    corridor_robot.absorb(teammate, corridor_robot.known.copy())
    assert corridor_robot.belief_version == version + 1
    assert step(corridor_robot) is None  # nothing left to search: idle instead of walking on


def test_absorb_without_news_keeps_the_route(corridor_robot: Robot) -> None:
    """Merging information the robot already has does not invalidate its route."""
    step(corridor_robot)
    version = corridor_robot.belief_version
    corridor_robot.absorb(corridor_robot.trail.copy(), corridor_robot.known.copy())
    assert corridor_robot.belief_version == version


def test_vectorised_crowding_matches_scalar_version() -> None:
    """crowding_many is exactly the per-cell crowding formula."""
    cells = [(0, 0), (3, 4), (7, 1)]
    others = [(1, 1), (5, 5), (0, 9)]
    np.testing.assert_allclose(crowding_many(cells, others), [crowding(c, others) for c in cells])
    assert crowding_many(cells, []).tolist() == [0.0, 0.0, 0.0]


def test_vectorised_perception_matches_python_loop() -> None:
    """Passing a positions array gives the same result as recomputing it."""
    mission = MissionControl(SwarmConfig(seed=3, num_agents=6))
    robots = mission.robots
    positions = np.asarray([r.pos for r in robots])
    for r in robots:
        assert perceive_teammates(r, robots, 5, positions) == perceive_teammates(r, robots, 5)


def test_route_cache_reduces_bfs_calls_without_changing_safety() -> None:
    """On a large mission, far fewer BFS searches run than route decisions are made, with zero collisions."""
    mission = MissionControl(SwarmConfig(grid_size=30, num_agents=8, battery=300, seed=2))
    result = mission.run()
    bfs = sum(r.bfs_calls for r in mission.robots)
    hits = sum(r.route_cache_hits for r in mission.robots)
    assert hits > 0 and result.collisions == 0
    assert hits / (hits + bfs) > 0.1


def test_radio_merge_uses_absorb() -> None:
    """RadioLink merges through Robot.absorb (so belief versions stay correct)."""
    a, b = Robot(0, (0, 0), 5, 10), Robot(1, (0, 1), 5, 10)
    b.trail.deposit((4, 4))
    before = a.belief_version
    RadioLink(5).share_pheromone_trails([a, b])
    assert a.trail.is_visited((4, 4)) and a.belief_version == before + 1


def test_benchmark_latency_within_real_time_budget() -> None:
    """scripts/benchmark.py measures per-tick latency; even 16 robots stay far below 50 ms."""
    from scripts.benchmark import measure

    row = measure(20, 16, 2, [1])
    assert row.ticks > 0 and 0.0 < row.mean_ms <= row.p95_ms <= row.max_ms
    assert row.max_ms < SwarmConfig().latency_budget_ms
    assert "| 20x20 | 16 | 2 |" in row.markdown()
