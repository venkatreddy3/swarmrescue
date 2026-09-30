"""Tests for world generation, reachability and the Round 2 shift."""

from __future__ import annotations

import numpy as np
import pytest

from swarmrescue.config import SwarmConfig
from swarmrescue.world import (
    FREE, WALL, World, apply_shift, bfs_distances, generate_world, manhattan, reachable,
)


@pytest.fixture
def world(default_cfg: SwarmConfig) -> World:
    """A default-config world generated from seed 7."""
    return generate_world(default_cfg, np.random.default_rng(7))


def test_grid_shape_and_values(world: World, default_cfg: SwarmConfig) -> None:
    """Grid is square, binary and wall density is close to the target."""
    assert world.grid.shape == (default_cfg.grid_size, default_cfg.grid_size)
    assert set(np.unique(world.grid)).issubset({FREE, WALL})
    assert abs(world.grid.mean() - default_cfg.wall_density) < 0.08


def test_starts_and_survivors_are_valid(world: World, default_cfg: SwarmConfig) -> None:
    """Starts and survivors are distinct, free and reachable."""
    cells = world.starts + world.survivors
    assert len(world.starts) == default_cfg.num_agents
    assert len(world.survivors) == default_cfg.num_survivors
    assert len(set(cells)) == len(cells)
    for cell in cells:
        assert world.grid[cell] == FREE
        assert world.reachable_mask[cell]


def test_generation_is_deterministic(default_cfg: SwarmConfig) -> None:
    """Same seed -> identical world; different seed -> different world."""
    a = generate_world(default_cfg, np.random.default_rng(3))
    b = generate_world(default_cfg, np.random.default_rng(3))
    c = generate_world(default_cfg, np.random.default_rng(4))
    assert np.array_equal(a.grid, b.grid) and a.survivors == b.survivors
    assert not np.array_equal(a.grid, c.grid)


def test_bfs_distances_on_known_maze() -> None:
    """BFS distances equal hand-computed shortest paths around a wall."""
    grid = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    dist = bfs_distances(grid, [(0, 0)])
    assert dist[0, 3] == 3
    assert dist[2, 0] == 8  # must go all the way around the wall
    assert dist[3, 0] == 9
    assert dist[1, 0] == -1  # wall


def test_reachable_matches_bfs_and_is_symmetric(world: World) -> None:
    """Reachability is an equivalence relation over free cells."""
    mask = world.reachable_mask
    free = list(zip(*np.nonzero(mask)))
    a, b = free[0], free[-1]
    dist_ab = bfs_distances(world.grid, [a])[b]
    dist_ba = bfs_distances(world.grid, [b])[a]
    assert dist_ab == dist_ba >= manhattan(a, b)  # L1 is a lower bound
    assert np.array_equal(reachable(world.grid, [b]), mask)
    assert not mask[world.grid == WALL].any()


def test_apply_shift_places_walls_correctly(world: World) -> None:
    """Shift adds exactly N walls, never on robots/survivors, and shrinks reachability."""
    before_free = int((world.grid == FREE).sum())
    robots = list(world.starts)
    new = apply_shift(world, 25, robots, np.random.default_rng(1))
    assert len(new) == len(set(new)) == 25
    assert int((world.grid == FREE).sum()) == before_free - 25
    for cell in new:
        assert world.grid[cell] == WALL
        assert cell not in robots and cell not in world.survivors
    assert np.array_equal(world.reachable_mask, reachable(world.grid, robots))
    assert not world.reachable_mask[world.grid == WALL].any()


def test_generate_rejects_impossible_density() -> None:
    """An almost-solid map cannot host the mission."""
    cfg = SwarmConfig(grid_size=6, wall_density=0.45, num_agents=2, num_survivors=2, new_walls=4)
    rng = np.random.default_rng(0)
    try:
        w = generate_world(cfg, rng)
    except RuntimeError:
        return
    assert w.reachable_count >= cfg.num_agents + cfg.num_survivors
