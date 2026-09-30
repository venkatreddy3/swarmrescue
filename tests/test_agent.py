"""Tests for robot decision-making, pheromone trails, BFS rerouting and the radio link."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from swarmrescue.agent import (
    DEADLOCK_TICKS,
    MODE_BFS,
    MODE_DEADLOCK,
    MODE_IDLE,
    MODE_PHEROMONE,
    UNKNOWN,
    PheromoneTrail,
    Robot,
    bfs_path,
    crowding,
)
from swarmrescue.config import SwarmConfig
from swarmrescue.coordination import RadioLink, perceive_teammates
from swarmrescue.world import DEBRIS, FREE, cells_where, neighbors4


def reference_distance(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Independent shortest-path length via Bellman-Ford style relaxation."""
    n = passable.shape[0]
    inf = 10**9
    dist = np.full((n, n), inf, dtype=np.int64)
    dist[start] = 0
    changed = True
    while changed:
        changed = False
        for r in range(n):
            for c in range(n):
                if not passable[r, c] or dist[r, c] == inf:
                    continue
                for nb in neighbors4((r, c), n):
                    if passable[nb] and dist[nb] > dist[r, c] + 1:
                        dist[nb] = dist[r, c] + 1
                        changed = True
    return int(dist[goal]) if dist[goal] < inf else -1


def mark_all_searched(robot: Robot, amount: float = 1.0) -> None:
    """Lay pheromone on every cell of the robot's trail."""
    robot.trail.intensity[:] = amount
    robot.trail.visited[:] = True


def unmark(robot: Robot, cell: tuple[int, int]) -> None:
    """Erase the pheromone on one cell (make it unsearched)."""
    robot.trail.intensity[cell] = 0.0
    robot.trail.visited[cell] = False


@pytest.fixture
def open_robot() -> Robot:
    """A robot at the centre of a fully known, empty 7x7 room."""
    robot = Robot(0, (3, 3), 7, battery=50)
    robot.known[:] = FREE
    return robot


@pytest.fixture
def zero_noise_cfg() -> SwarmConfig:
    """Config with no randomness so decisions are exact."""
    return SwarmConfig(randomness=0.0, pheromone_weight=1.0, spread_weight=1.0)


def test_pheromone_trail_deposit_and_merge() -> None:
    """Deposits accumulate; merge takes the element-wise max and unions visits."""
    a, b = PheromoneTrail(4), PheromoneTrail(4)
    a.deposit((1, 1))
    a.deposit((1, 1))
    b.deposit((1, 1))
    b.deposit((2, 2), 3.0)
    assert a.strength((1, 1)) == 2.0 and a.is_visited((1, 1)) and not a.is_visited((2, 2))
    a.merge(b)
    assert a.strength((1, 1)) == 2.0 and a.strength((2, 2)) == 3.0 and a.is_visited((2, 2))
    clone = a.copy()
    clone.deposit((0, 0))
    assert not a.is_visited((0, 0))


@pytest.mark.parametrize("seed", range(12))
def test_bfs_returns_true_shortest_path(seed: int) -> None:
    """BFS path length equals an independently computed shortest distance."""
    rng = np.random.default_rng(seed)
    passable = rng.random((9, 9)) > 0.3
    free = cells_where(passable)
    start, goal = free[0], free[-1]
    goal_mask = np.zeros_like(passable)
    goal_mask[goal] = True
    path = bfs_path(passable, start, goal_mask)
    expected = reference_distance(passable, start, goal)
    if expected < 0:
        assert path is None
        return
    assert path is not None
    assert len(path) - 1 == expected
    assert path[0] == start and path[-1] == goal
    for a, b in itertools.pairwise(path):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        assert passable[b]


def test_bfs_reroutes_around_blocked_cells() -> None:
    """A blocked cell forces a detour that is exactly 2 steps longer."""
    passable = np.ones((5, 5), dtype=bool)
    goal = np.zeros_like(passable)
    goal[2, 4] = True
    direct = bfs_path(passable, (2, 0), goal)
    detour = bfs_path(passable, (2, 0), goal, blocked={(2, 2)})
    assert direct is not None and detour is not None
    assert len(direct) - 1 == 4 and len(detour) - 1 == 6
    assert (2, 2) not in detour


def test_crowding_formula() -> None:
    """Crowding is the exact sum of 1/(1+L1 distance)."""
    assert crowding((0, 0), [(0, 1), (2, 2)]) == pytest.approx(1 / 2 + 1 / 5)
    assert crowding((0, 0), []) == 0.0


def test_pheromone_rule_picks_unvisited(open_robot: Robot, zero_noise_cfg: SwarmConfig) -> None:
    """With zero noise the robot picks the only unsearched neighbour."""
    for nb in [(2, 3), (4, 3), (3, 2)]:
        open_robot.trail.deposit(nb, 2.0)
    move = open_robot.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0))
    assert move == (3, 4) and open_robot.mode == MODE_PHEROMONE


def test_spread_term_avoids_teammates(open_robot: Robot, zero_noise_cfg: SwarmConfig) -> None:
    """Between two unsearched cells, the one farther from a teammate wins."""
    for nb in [(2, 3), (4, 3)]:
        open_robot.trail.deposit(nb, 5.0)
    teammate = (3, 6)  # east side
    move = open_robot.choose_move(zero_noise_cfg, {teammate}, [teammate], np.random.default_rng(0))
    assert move == (3, 2)


def test_bfs_reroute_when_neighbours_searched(open_robot: Robot, zero_noise_cfg: SwarmConfig) -> None:
    """If every neighbour is searched, the first step heads to the nearest unsearched cell."""
    mark_all_searched(open_robot)
    unmark(open_robot, (3, 6))  # target three cells to the east
    move = open_robot.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0))
    assert move == (3, 4) and open_robot.mode == MODE_BFS


def test_idle_when_everything_searched(open_robot: Robot, zero_noise_cfg: SwarmConfig) -> None:
    """With no unsearched known cell, the robot saves battery and stays."""
    mark_all_searched(open_robot)
    assert open_robot.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0)) is None
    assert open_robot.mode == MODE_IDLE


def test_never_moves_into_debris_or_robots(zero_noise_cfg: SwarmConfig) -> None:
    """Across many random situations, chosen cells are free and unoccupied (collision-free)."""
    rng = np.random.default_rng(11)
    for _ in range(300):
        grid = (rng.random((8, 8)) < 0.3).astype(np.int8)
        start = (int(rng.integers(8)), int(rng.integers(8)))
        grid[start] = FREE
        robot = Robot(0, start, 8, battery=10)
        robot.sense_debris(grid, 1)
        robot.trail.intensity = rng.integers(0, 3, size=(8, 8)).astype(np.float64)
        robot.trail.visited = robot.trail.intensity > 0
        robot.stuck_ticks = int(rng.integers(0, 5))
        blocked = {nb for nb in neighbors4(start, 8) if rng.random() < 0.3}
        move = robot.choose_move(zero_noise_cfg, blocked, list(blocked), rng)
        if move is not None:
            assert grid[move] == FREE
            assert move not in blocked
            assert abs(move[0] - start[0]) + abs(move[1] - start[1]) == 1


def test_deadlock_breaker_after_three_blocked_ticks(open_robot: Robot, zero_noise_cfg: SwarmConfig) -> None:
    """Three blocked ticks trigger a random free move."""
    mark_all_searched(open_robot)
    unmark(open_robot, (0, 0))
    wall_of_robots = {(2, 3), (3, 2)}  # the two cells toward the goal
    ring = {(4, 3), (3, 4)}
    for _ in range(DEADLOCK_TICKS):
        open_robot.mode = "blocked"
        open_robot.commit(None)
    assert open_robot.stuck_ticks == DEADLOCK_TICKS
    move = open_robot.choose_move(zero_noise_cfg, wall_of_robots, [], np.random.default_rng(0))
    assert open_robot.mode == MODE_DEADLOCK and move in ring
    open_robot.commit(move)
    assert open_robot.stuck_ticks == 0


def test_failed_or_empty_robot_never_moves(open_robot: Robot, zero_noise_cfg: SwarmConfig) -> None:
    """Failed robots and robots with a flat battery return no move."""
    open_robot.fail()
    assert open_robot.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0)) is None
    other = Robot(1, (1, 1), 5, battery=1)
    other.known[:] = FREE
    other.commit((1, 2))
    assert other.battery == 0 and other.moves == 1
    assert other.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0)) is None


def test_reroute_after_new_debris() -> None:
    """Sensing new debris on the planned route changes the next move."""
    cfg = SwarmConfig(randomness=0.0)
    grid = np.zeros((5, 5), dtype=np.int8)
    robot = Robot(0, (2, 0), 5, battery=50)
    robot.known[:] = FREE
    mark_all_searched(robot)
    unmark(robot, (2, 4))
    assert robot.choose_move(cfg, set(), [], np.random.default_rng(0)) == (2, 1)
    grid[2, 1] = DEBRIS  # debris falls on the route; the robot is not told
    surprises = robot.sense_debris(grid, 1)
    assert surprises == 1 and robot.known[2, 1] == DEBRIS
    move = robot.choose_move(cfg, set(), [], np.random.default_rng(0))
    assert move in {(1, 0), (3, 0)}


def test_sense_updates_window_only() -> None:
    """Sensing reveals exactly the (2r+1)^2 window around the robot."""
    grid = np.zeros((7, 7), dtype=np.int8)
    robot = Robot(0, (3, 3), 7, battery=5)
    robot.sense_debris(grid, 1)
    assert int((robot.known != UNKNOWN).sum()) == 9


def test_radio_link_merges_with_maximum() -> None:
    """In-range robots end up with the element-wise max of their trails."""
    a, b, far = Robot(0, (0, 0), 6, 10), Robot(1, (0, 3), 6, 10), Robot(2, (5, 5), 6, 10)
    a.trail.deposit((1, 1), 4.0)
    b.trail.deposit((1, 1), 2.0)
    b.trail.deposit((2, 2), 7.0)
    a.known[4, 4] = DEBRIS
    expected = np.maximum(a.trail.intensity, b.trail.intensity)
    links = RadioLink(comm_range=5).share_pheromone_trails([a, b, far])
    assert links == 1
    assert np.array_equal(a.trail.intensity, expected) and np.array_equal(b.trail.intensity, expected)
    assert b.known[4, 4] == DEBRIS
    assert far.trail.strength((2, 2)) == 0.0  # out of range: nothing shared


def test_failed_robot_does_not_transmit() -> None:
    """A failed robot neither sends nor receives trails."""
    a, b = Robot(0, (0, 0), 5, 10), Robot(1, (0, 1), 5, 10)
    b.trail.deposit((3, 3), 9.0)
    a.fail()
    assert RadioLink(comm_range=5).share_pheromone_trails([a, b]) == 0
    assert a.trail.strength((3, 3)) == 0.0


def test_radio_link_cut_stops_sharing() -> None:
    """After the radio link is cut, nothing is shared and perception shrinks to 2."""
    a, b = Robot(0, (0, 0), 5, 10), Robot(1, (0, 1), 5, 10)
    b.trail.deposit((3, 3))
    radio = RadioLink(comm_range=5)
    assert radio.perception_radius == 5
    radio.cut()
    assert not radio.online and radio.perception_radius == 2
    assert radio.share_pheromone_trails([a, b]) == 0
    assert not a.trail.is_visited((3, 3))


def test_perception_includes_failed_robots_as_obstacles() -> None:
    """Failed robots block cells but do not count toward crowding."""
    me, dead, near, far = (Robot(i, p, 10, 10) for i, p in enumerate([(0, 0), (0, 1), (1, 1), (9, 9)]))
    dead.fail()
    radio = RadioLink(5)
    radio.cut()
    blocked, others = perceive_teammates(me, [me, dead, near, far], radio.perception_radius)
    assert blocked == {(0, 1), (1, 1)}
    assert others == [(1, 1)]
