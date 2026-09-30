"""Disaster-zone world model: map generation, reachability and scenario shift.

The building is an occupancy grid where ``0`` is free space and ``1`` is
debris or wall. Robots enter through the top-left corner (the "entrance").
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from swarmrescue.config import SwarmConfig

Cell = tuple[int, int]

FREE: int = 0
WALL: int = 1
DIRECTIONS: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ENTRANCE: Cell = (0, 0)
_MAX_GENERATION_ATTEMPTS: int = 200


@dataclass
class World:
    """Ground-truth state of the disaster zone.

    Attributes:
        grid: ``int8`` occupancy grid (0 free, 1 wall). Mutated by the shift.
        starts: Initial robot cells (distinct, free, connected to entrance).
        survivors: Cells where survivors are trapped.
        reachable_mask: Boolean mask of free cells the swarm can reach.
        new_walls: Cells converted to debris by the Round 2 shift.
    """

    grid: np.ndarray
    starts: list[Cell]
    survivors: list[Cell]
    reachable_mask: np.ndarray
    new_walls: list[Cell] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Side length of the square grid."""
        return int(self.grid.shape[0])

    @property
    def reachable_count(self) -> int:
        """Number of reachable free cells (coverage denominator)."""
        return int(self.reachable_mask.sum())


def in_bounds(cell: Cell, size: int) -> bool:
    """Return True if ``cell`` lies inside a ``size`` x ``size`` grid."""
    return 0 <= cell[0] < size and 0 <= cell[1] < size


def neighbors4(cell: Cell, size: int) -> Iterator[Cell]:
    """Yield the in-bounds 4-connected neighbours of ``cell``."""
    r, c = cell
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield (nr, nc)


def manhattan(a: Cell, b: Cell) -> int:
    """Manhattan (L1) distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_distances(grid: np.ndarray, sources: Iterable[Cell]) -> np.ndarray:
    """Multi-source BFS distances over free cells.

    Args:
        grid: Occupancy grid (0 free, non-zero blocked).
        sources: Start cells; blocked sources are ignored.

    Returns:
        ``int`` array of shortest 4-connected path lengths, ``-1`` where a
        cell is blocked or unreachable.
    """
    size = grid.shape[0]
    free = (grid == FREE).tolist()
    dist = np.full(grid.shape, -1, dtype=int)
    queue: deque[Cell] = deque()
    for src in sources:
        if in_bounds(src, size) and free[src[0]][src[1]] and dist[src] < 0:
            dist[src] = 0
            queue.append(src)
    while queue:
        cell = queue.popleft()
        d = dist[cell] + 1
        for nb in neighbors4(cell, size):
            if free[nb[0]][nb[1]] and dist[nb] < 0:
                dist[nb] = d
                queue.append(nb)
    return dist


def reachable(grid: np.ndarray, sources: Iterable[Cell]) -> np.ndarray:
    """Boolean mask of free cells 4-connected to any cell in ``sources``."""
    return bfs_distances(grid, sources) >= 0


def generate_world(cfg: SwarmConfig, rng: np.random.Generator) -> World:
    """Create a random collapsed-building map for ``cfg``.

    Walls are sampled i.i.d. with probability ``cfg.wall_density``. The
    entrance corner is cleared, and maps whose reachable region is too small
    to host the robots and survivors (or less than 40% of the free space)
    are rejected and resampled.

    Args:
        cfg: Mission configuration.
        rng: Seeded random generator (the only randomness source).

    Returns:
        A fully initialised :class:`World`.

    Raises:
        RuntimeError: If no valid map is found after many attempts.
    """
    n = cfg.grid_size
    needed = cfg.num_agents + cfg.num_survivors + 1
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = (rng.random((n, n)) < cfg.wall_density).astype(np.int8)
        grid[0:2, 0:2] = FREE  # clear the entrance
        dist = bfs_distances(grid, [ENTRANCE])
        mask = dist >= 0
        free_total = int((grid == FREE).sum())
        if mask.sum() < max(needed, 0.4 * free_total):
            continue
        # Robots start at the reachable cells closest to the entrance.
        order = sorted(zip(*np.nonzero(mask)), key=lambda rc: (dist[rc], rc))
        starts = [(int(r), int(c)) for r, c in order[: cfg.num_agents]]
        # Survivors hide in reachable cells away from the robots.
        candidates = [(int(r), int(c)) for r, c in order[cfg.num_agents:]]
        picks = rng.choice(len(candidates), size=cfg.num_survivors, replace=False)
        survivors = [candidates[int(i)] for i in sorted(picks)]
        return World(grid=grid, starts=starts, survivors=survivors, reachable_mask=mask)
    raise RuntimeError("could not generate a valid world; lower wall_density")


def apply_shift(
    world: World,
    num_new_walls: int,
    robot_cells: Iterable[Cell],
    rng: np.random.Generator,
) -> list[Cell]:
    """Round 2 aftershock: drop new debris onto free cells.

    New walls never land on robots or survivors. The world's reachable mask
    is recomputed from the robots' positions (cells still reachable by the
    swarm). Robots are *not* told about the new walls; they must sense them.

    Args:
        world: World to mutate in place.
        num_new_walls: How many free cells become debris.
        robot_cells: Current robot positions (protected from debris).
        rng: Seeded random generator.

    Returns:
        The list of cells that became walls.
    """
    robots = set(robot_cells)
    protected = robots | set(world.survivors)
    free_cells = [
        (int(r), int(c)) for r, c in zip(*np.nonzero(world.grid == FREE))
        if (int(r), int(c)) not in protected
    ]
    count = min(num_new_walls, len(free_cells))
    picks = rng.choice(len(free_cells), size=count, replace=False) if count else []
    new_walls = [free_cells[int(i)] for i in sorted(picks)]
    for cell in new_walls:
        world.grid[cell] = WALL
    world.new_walls.extend(new_walls)
    world.reachable_mask = reachable(world.grid, robots)
    return new_walls
