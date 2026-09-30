"""Tests for robot decision-making, BFS planning and map sharing."""

from __future__ import annotations

import numpy as np
import pytest

from swarmrescue.agent import (
    DEADLOCK_TICKS, MODE_BFS, MODE_DEADLOCK, MODE_IDLE, MODE_PHEROMONE, UNKNOWN, Agent, bfs_path, crowding,
)
from swarmrescue.config import SwarmConfig
from swarmrescue.coordination import perceive, perception_radius, share_maps
from swarmrescue.world import FREE, WALL, neighbors4


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


@pytest.fixture
def open_agent() -> Agent:
    """A robot at the centre of a fully known, empty 7x7 room."""
    agent = Agent(0, (3, 3), 7, battery=50)
    agent.known[:] = FREE
    return agent


@pytest.fixture
def zero_noise_cfg() -> SwarmConfig:
    """Config with no randomness so decisions are exact."""
    return SwarmConfig(randomness=0.0, pheromone_weight=1.0, spread_weight=1.0)


@pytest.mark.parametrize("seed", range(12))
def test_bfs_returns_true_shortest_path(seed: int) -> None:
    """BFS path length equals an independently computed shortest distance."""
    rng = np.random.default_rng(seed)
    passable = rng.random((9, 9)) > 0.3
    free = list(zip(*np.nonzero(passable)))
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
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        assert passable[b]


def test_bfs_routes_around_blocked_cells() -> None:
    """A blocked cell forces the detour and lengthens the path by exactly 2."""
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


def test_pheromone_rule_picks_unvisited(open_agent: Agent, zero_noise_cfg: SwarmConfig) -> None:
    """With zero noise the robot picks the only unvisited neighbour."""
    for nb in [(2, 3), (4, 3), (3, 2)]:
        open_agent.visits[nb] = 2
    move = open_agent.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0))
    assert move == (3, 4) and open_agent.mode == MODE_PHEROMONE


def test_spread_term_avoids_teammates(open_agent: Agent, zero_noise_cfg: SwarmConfig) -> None:
    """Between two unvisited cells, the one farther from a teammate wins."""
    for nb in [(2, 3), (4, 3)]:
        open_agent.visits[nb] = 5
    teammate = (3, 6)  # east side
    move = open_agent.choose_move(zero_noise_cfg, {teammate}, [teammate], np.random.default_rng(0))
    assert move == (3, 2)


def test_bfs_fallback_when_neighbours_visited(open_agent: Agent, zero_noise_cfg: SwarmConfig) -> None:
    """If every neighbour is visited, the first step heads to the nearest unvisited cell."""
    open_agent.visits[:] = 1
    open_agent.visits[3, 6] = 0  # target three cells to the east
    move = open_agent.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0))
    assert move == (3, 4) and open_agent.mode == MODE_BFS


def test_idle_when_everything_explored(open_agent: Agent, zero_noise_cfg: SwarmConfig) -> None:
    """With no unvisited known cell the robot saves energy and stays."""
    open_agent.visits[:] = 1
    assert open_agent.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0)) is None
    assert open_agent.mode == MODE_IDLE


def test_never_moves_into_walls_or_robots(zero_noise_cfg: SwarmConfig) -> None:
    """Across many random situations, chosen cells are free and unoccupied."""
    rng = np.random.default_rng(11)
    for _ in range(300):
        grid = (rng.random((8, 8)) < 0.3).astype(np.int8)
        start = (int(rng.integers(8)), int(rng.integers(8)))
        grid[start] = FREE
        agent = Agent(0, start, 8, battery=10)
        agent.sense(grid, 1)
        agent.visits = rng.integers(0, 3, size=(8, 8)).astype(np.int32)
        agent.stuck_ticks = int(rng.integers(0, 5))
        blocked = {nb for nb in neighbors4(start, 8) if rng.random() < 0.3}
        move = agent.choose_move(zero_noise_cfg, blocked, list(blocked), rng)
        if move is not None:
            assert grid[move] == FREE
            assert move not in blocked
            assert abs(move[0] - start[0]) + abs(move[1] - start[1]) == 1


def test_deadlock_breaker_after_three_blocked_ticks(open_agent: Agent, zero_noise_cfg: SwarmConfig) -> None:
    """Three blocked ticks trigger a random free move."""
    open_agent.visits[:] = 1
    open_agent.visits[0, 0] = 0
    wall_of_robots = {(2, 3), (3, 2)}  # the two cells toward the goal
    ring = {(4, 3), (3, 4)}
    for _ in range(DEADLOCK_TICKS):
        # BFS finds a detour here, so emulate a physical block via commit.
        open_agent.mode = "blocked"
        open_agent.commit(None)
    assert open_agent.stuck_ticks == DEADLOCK_TICKS
    move = open_agent.choose_move(zero_noise_cfg, wall_of_robots, [], np.random.default_rng(0))
    assert open_agent.mode == MODE_DEADLOCK and move in ring
    open_agent.commit(move)
    assert open_agent.stuck_ticks == 0


def test_dead_or_empty_robot_never_moves(open_agent: Agent, zero_noise_cfg: SwarmConfig) -> None:
    """Failed robots and robots with a flat battery return no move."""
    open_agent.fail()
    assert open_agent.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0)) is None
    other = Agent(1, (1, 1), 5, battery=1)
    other.known[:] = FREE
    other.commit((1, 2))
    assert other.battery == 0 and other.moves == 1
    assert other.choose_move(zero_noise_cfg, set(), [], np.random.default_rng(0)) is None


def test_replanning_after_new_wall() -> None:
    """Sensing new debris on the planned route changes the next move."""
    cfg = SwarmConfig(randomness=0.0)
    grid = np.zeros((5, 5), dtype=np.int8)
    agent = Agent(0, (2, 0), 5, battery=50)
    agent.known[:] = FREE
    agent.visits[:] = 1
    agent.visits[2, 4] = 0
    assert agent.choose_move(cfg, set(), [], np.random.default_rng(0)) == (2, 1)
    grid[2, 1] = WALL  # debris falls on the route; the robot is not told
    surprises = agent.sense(grid, 1)
    assert surprises == 1 and agent.known[2, 1] == WALL
    move = agent.choose_move(cfg, set(), [], np.random.default_rng(0))
    assert move in {(1, 0), (3, 0)}


def test_sense_updates_window_only() -> None:
    """Sensing reveals exactly the (2r+1)^2 window around the robot."""
    grid = np.zeros((7, 7), dtype=np.int8)
    agent = Agent(0, (3, 3), 7, battery=5)
    agent.sense(grid, 1)
    assert int((agent.known != UNKNOWN).sum()) == 9


def test_share_maps_merges_with_maximum() -> None:
    """In-range robots end up with the element-wise max of their maps."""
    a, b, far = Agent(0, (0, 0), 6, 10), Agent(1, (0, 3), 6, 10), Agent(2, (5, 5), 6, 10)
    a.visits[1, 1], b.visits[1, 1], b.visits[2, 2] = 4, 2, 7
    a.known[4, 4] = WALL
    expected_v = np.maximum(a.visits, b.visits)
    links = share_maps([a, b, far], comm_range=5)
    assert links == 1
    assert np.array_equal(a.visits, expected_v) and np.array_equal(b.visits, expected_v)
    assert b.known[4, 4] == WALL
    assert far.visits[2, 2] == 0  # out of range: nothing shared


def test_failed_robot_does_not_share() -> None:
    """A dead robot neither sends nor receives maps."""
    a, b = Agent(0, (0, 0), 5, 10), Agent(1, (0, 1), 5, 10)
    b.visits[3, 3] = 9
    a.fail()
    assert share_maps([a, b], comm_range=5) == 0
    assert a.visits[3, 3] == 0


def test_perception_includes_dead_as_obstacles() -> None:
    """Dead robots block cells but do not count toward crowding."""
    me, dead, near, far = (Agent(i, p, 10, 10) for i, p in enumerate([(0, 0), (0, 1), (1, 1), (9, 9)]))
    dead.fail()
    blocked, others = perceive(me, [me, dead, near, far], radius=perception_radius(False, 5))
    assert blocked == {(0, 1), (1, 1)}
    assert others == [(1, 1)]
    assert perception_radius(True, 5) == 5
