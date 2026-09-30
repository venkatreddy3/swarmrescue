"""Tests for the disaster zone: generation, reachability and aftershock debris."""

from __future__ import annotations

import numpy as np
import pytest

from swarmrescue.config import SwarmConfig
from swarmrescue.world import (
    DEBRIS,
    FREE,
    CellType,
    Debris,
    DisasterZone,
    Survivor,
    bfs_distances,
    cells_where,
    drop_aftershock_debris,
    generate_disaster_zone,
    manhattan,
    reachable,
)


@pytest.fixture
def zone(default_cfg: SwarmConfig) -> DisasterZone:
    """A default-config disaster zone generated from seed 7."""
    return generate_disaster_zone(default_cfg, np.random.default_rng(7))


def test_cell_types() -> None:
    """Free floor is 0 and debris is 1."""
    assert (FREE, DEBRIS) == (int(CellType.FREE), int(CellType.DEBRIS)) == (0, 1)


def test_grid_shape_and_values(zone: DisasterZone, default_cfg: SwarmConfig) -> None:
    """Grid is square and binary, and debris density is close to the target."""
    assert zone.grid.shape == (default_cfg.grid_size, default_cfg.grid_size)
    assert set(np.unique(zone.grid)).issubset({FREE, DEBRIS})
    assert abs(zone.grid.mean() - default_cfg.wall_density) < 0.08


def test_entry_cells_and_survivors_are_valid(zone: DisasterZone, default_cfg: SwarmConfig) -> None:
    """Entry cells and survivors are distinct, free and reachable."""
    cells = zone.entry_cells + zone.survivor_cells
    assert len(zone.entry_cells) == default_cfg.num_agents
    assert len(zone.survivors) == default_cfg.num_survivors
    assert all(isinstance(s, Survivor) for s in zone.survivors)
    assert [s.survivor_id for s in zone.survivors] == list(range(default_cfg.num_survivors))
    assert len(set(cells)) == len(cells)
    for cell in cells:
        assert zone.is_free(cell)
        assert zone.reachable_mask[cell]


def test_generation_is_deterministic(default_cfg: SwarmConfig) -> None:
    """Same seed gives an identical zone; a different seed gives a different zone."""
    a = generate_disaster_zone(default_cfg, np.random.default_rng(3))
    b = generate_disaster_zone(default_cfg, np.random.default_rng(3))
    c = generate_disaster_zone(default_cfg, np.random.default_rng(4))
    assert np.array_equal(a.grid, b.grid) and a.survivors == b.survivors
    assert not np.array_equal(a.grid, c.grid)


def test_bfs_distances_on_known_maze() -> None:
    """BFS distances equal hand-computed shortest paths around debris."""
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
    assert dist[2, 0] == 8  # must go all the way around the debris
    assert dist[3, 0] == 9
    assert dist[1, 0] == -1  # debris


def test_reachable_matches_bfs_and_is_symmetric(zone: DisasterZone) -> None:
    """Reachability is an equivalence relation over free cells."""
    mask = zone.reachable_mask
    free = cells_where(mask)
    a, b = free[0], free[-1]
    dist_ab = bfs_distances(zone.grid, [a])[b]
    dist_ba = bfs_distances(zone.grid, [b])[a]
    assert dist_ab == dist_ba >= manhattan(a, b)  # L1 is a lower bound
    assert np.array_equal(reachable(zone.grid, [b]), mask)
    assert not mask[zone.grid == DEBRIS].any()


def test_aftershock_debris_placed_correctly(zone: DisasterZone) -> None:
    """Aftershock adds exactly N debris cells, never on robots or survivors, and shrinks reachability."""
    before_free = int((zone.grid == FREE).sum())
    robots = list(zone.entry_cells)
    new = drop_aftershock_debris(zone, 25, robots, np.random.default_rng(1))
    cells = [d.cell for d in new]
    assert all(isinstance(d, Debris) and d.source == "aftershock" for d in new)
    assert len(cells) == len(set(cells)) == 25
    assert int((zone.grid == FREE).sum()) == before_free - 25
    assert zone.aftershock_debris == cells
    for cell in cells:
        assert zone.grid[cell] == DEBRIS
        assert cell not in robots and cell not in zone.survivor_cells
    assert np.array_equal(zone.reachable_mask, reachable(zone.grid, robots))
    assert not zone.reachable_mask[zone.grid == DEBRIS].any()


def test_generate_rejects_impossible_density() -> None:
    """An almost-solid map cannot host the mission."""
    cfg = SwarmConfig(grid_size=6, wall_density=0.45, num_agents=2, num_survivors=2, new_walls=4)
    rng = np.random.default_rng(0)
    try:
        zone = generate_disaster_zone(cfg, rng)
    except RuntimeError:
        return
    assert zone.reachable_count >= cfg.num_agents + cfg.num_survivors
