"""Disaster-zone model for search and rescue.

The collapsed building is a :class:`DisasterZone`: an occupancy grid where
``FREE`` (0) is open floor and ``DEBRIS`` (1) is rubble or wall. Rescue robots
enter through the top-left entry point. :class:`Survivor` objects are trapped
in reachable cells. In Round 2 an :class:`Aftershock` drops new
:class:`Debris` that the robots are not told about.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from swarmrescue.config import SwarmConfig

Cell = tuple[int, int]


class CellType(IntEnum):
    """Occupancy value of one disaster-zone cell."""

    FREE = 0
    DEBRIS = 1


FREE: int = int(CellType.FREE)
DEBRIS: int = int(CellType.DEBRIS)
WALL: int = DEBRIS  # walls and debris are equally impassable
DIRECTIONS: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ENTRY_POINT: Cell = (0, 0)
_MAX_GENERATION_ATTEMPTS: int = 200


@dataclass(frozen=True)
class Survivor:
    """A trapped person waiting for rescue.

    Attributes:
        survivor_id: Index of the survivor.
        cell: Grid cell where the survivor is trapped.
    """

    survivor_id: int
    cell: Cell


@dataclass(frozen=True)
class Debris:
    """One impassable rubble cell.

    Attributes:
        cell: Location of the rubble.
        source: ``"collapse"`` (initial map) or ``"aftershock"`` (Round 2).
    """

    cell: Cell
    source: str = "collapse"


@dataclass(frozen=True)
class Aftershock:
    """Record of the Round 2 scenario shift.

    Attributes:
        tick: Tick at which the aftershock struck.
        debris: New debris dropped (robots are not told where).
        failed_robot_id: Robot knocked out (the single point of failure test).
        radio_lost: Whether the radio link went down.
    """

    tick: int
    debris: tuple[Debris, ...]
    failed_robot_id: int | None
    radio_lost: bool


@dataclass
class DisasterZone:
    """Ground truth of the collapsed building.

    Attributes:
        grid: ``int8`` occupancy grid (0 free, 1 debris). Mutated by aftershocks.
        entry_cells: Starting cells of the rescue robots (distinct and free).
        survivors: Trapped survivors.
        reachable_mask: Boolean mask of free cells the swarm can reach
            (the coverage denominator).
        aftershock_debris: Cells turned into debris by aftershocks.
    """

    grid: np.ndarray
    entry_cells: list[Cell]
    survivors: list[Survivor]
    reachable_mask: np.ndarray
    aftershock_debris: list[Cell] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Side length of the square grid."""
        return int(self.grid.shape[0])

    @property
    def reachable_count(self) -> int:
        """Number of reachable free cells (coverage denominator)."""
        return int(self.reachable_mask.sum())

    @property
    def survivor_cells(self) -> list[Cell]:
        """Cells of all survivors, in survivor-id order."""
        return [s.cell for s in self.survivors]

    def is_free(self, cell: Cell) -> bool:
        """True if ``cell`` is inside the zone and not debris."""
        return in_bounds(cell, self.size) and self.grid[cell] == FREE


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


def cells_where(mask: np.ndarray) -> list[Cell]:
    """Cells where ``mask`` is True, in row-major order (vectorised with ``np.argwhere``)."""
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def bfs_distances(grid: np.ndarray, sources: Iterable[Cell]) -> np.ndarray:
    """Multi-source BFS distances over free cells.

    Args:
        grid: Occupancy grid (0 free, non-zero debris).
        sources: Start cells; sources on debris are ignored.

    Returns:
        ``int`` array of shortest 4-connected path lengths, ``-1`` where a
        cell is debris or unreachable.
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


def generate_disaster_zone(cfg: SwarmConfig, rng: np.random.Generator) -> DisasterZone:
    """Create a random collapsed building for a search-and-rescue mission.

    Debris is sampled independently in each cell with probability
    ``cfg.wall_density``. The entry point is cleared. A map is rejected and
    resampled if its reachable region is too small to hold the robots and
    survivors, or smaller than 40% of the free space.

    Args:
        cfg: Mission configuration.
        rng: Seeded random generator (the only randomness source).

    Returns:
        A fully initialised :class:`DisasterZone`.

    Raises:
        RuntimeError: If no valid map is found after many attempts.
    """
    n = cfg.grid_size
    needed = cfg.num_agents + cfg.num_survivors + 1
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = (rng.random((n, n)) < cfg.wall_density).astype(np.int8)
        grid[0:2, 0:2] = FREE  # clear the entry point
        dist = bfs_distances(grid, [ENTRY_POINT])
        mask = dist >= 0
        free_total = int((grid == FREE).sum())
        if mask.sum() < max(needed, 0.4 * free_total):
            continue
        # Robots start at the reachable cells closest to the entry point.
        order = sorted(cells_where(mask), key=lambda rc: (dist[rc], rc))
        entry_cells = [(int(r), int(c)) for r, c in order[: cfg.num_agents]]
        # Survivors are trapped in reachable cells away from the robots.
        candidates = [(int(r), int(c)) for r, c in order[cfg.num_agents :]]
        picks = rng.choice(len(candidates), size=cfg.num_survivors, replace=False)
        survivors = [Survivor(i, candidates[int(p)]) for i, p in enumerate(sorted(picks))]
        return DisasterZone(grid=grid, entry_cells=entry_cells, survivors=survivors, reachable_mask=mask)
    raise RuntimeError("could not generate a valid disaster zone; lower wall_density")


def drop_aftershock_debris(
    zone: DisasterZone,
    num_debris: int,
    robot_cells: Iterable[Cell],
    rng: np.random.Generator,
) -> list[Debris]:
    """Aftershock: turn random free cells into debris.

    New debris never lands on a robot or a survivor. The zone's reachable
    mask is recomputed from the robots' positions. Robots are *not* told
    where the debris fell; they detect it with their sensors and reroute.

    Args:
        zone: Disaster zone to mutate in place.
        num_debris: How many free cells become debris.
        robot_cells: Current robot positions (protected from debris).
        rng: Seeded random generator.

    Returns:
        The new :class:`Debris` records.
    """
    robots = set(robot_cells)
    protected = robots | set(zone.survivor_cells)
    free_cells = [cell for cell in cells_where(zone.grid == FREE) if cell not in protected]
    count = min(num_debris, len(free_cells))
    picks: list[int] = sorted(int(i) for i in rng.choice(len(free_cells), size=count, replace=False)) if count else []
    new_cells = [free_cells[i] for i in picks]
    for cell in new_cells:
        zone.grid[cell] = DEBRIS
    zone.aftershock_debris.extend(new_cells)
    zone.reachable_mask = reachable(zone.grid, robots)
    return [Debris(cell, "aftershock") for cell in new_cells]
