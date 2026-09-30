"""Autonomous rescue robot: local sensing, pheromone scoring and BFS fallback.

Each robot owns two private maps:

* ``known``  - its belief of the building (-1 unknown, 0 free, 1 wall),
  updated by its short-range wall sensor and by merges with teammates;
* ``visits`` - its ant-style pheromone map: how often each cell has been
  visited (by itself or, after merging, by teammates it met).

A robot never sees the ground-truth grid directly; it decides every move from
these maps plus the positions of teammates it can currently perceive.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from swarmrescue.config import SwarmConfig
from swarmrescue.world import FREE, WALL, Cell, manhattan, neighbors4

UNKNOWN: int = -1
DEADLOCK_TICKS: int = 3

MODE_PHEROMONE = "pheromone"
MODE_BFS = "bfs"
MODE_DEADLOCK = "deadlock"
MODE_BLOCKED = "blocked"
MODE_IDLE = "idle"
MODE_OFF = "off"


def bfs_path(
    passable: np.ndarray,
    start: Cell,
    goal_mask: np.ndarray,
    blocked: Iterable[Cell] = (),
) -> list[Cell] | None:
    """Breadth-first search to the nearest goal cell.

    Args:
        passable: Boolean grid, True where the robot may drive.
        start: Starting cell (always expanded, even if not passable).
        goal_mask: Boolean grid, True at acceptable destination cells.
        blocked: Extra cells treated as obstacles (e.g. other robots).

    Returns:
        The shortest 4-connected path ``[start, ..., goal]`` to the closest
        goal cell other than ``start``, or ``None`` if no goal is reachable.
    """
    size = passable.shape[0]
    ok = passable.tolist()
    goals = goal_mask.tolist()
    for r, c in blocked:
        if 0 <= r < size and 0 <= c < size:
            ok[r][c] = False
    parent: dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])
    while queue:
        cell = queue.popleft()
        for nb in neighbors4(cell, size):
            if nb in parent or not ok[nb[0]][nb[1]]:
                continue
            parent[nb] = cell
            if goals[nb[0]][nb[1]]:
                path = [nb]
                prev = parent[nb]
                while prev is not None:
                    path.append(prev)
                    prev = parent[prev]
                return path[::-1]
            queue.append(nb)
    return None


def crowding(cell: Cell, others: Iterable[Cell]) -> float:
    """Crowding penalty: sum of ``1 / (1 + manhattan)`` to perceived robots."""
    return float(sum(1.0 / (1.0 + manhattan(cell, o)) for o in others))


class Agent:
    """A decentralized rescue robot.

    Attributes:
        agent_id: Integer identifier (robot 0 fails in Round 2).
        pos: Current cell.
        battery: Remaining moves.
        alive: False once the robot has suffered a hardware failure.
        visits: Private pheromone (visit-count) map.
        known: Private occupancy belief (-1 unknown, 0 free, 1 wall).
        stuck_ticks: Consecutive ticks the robot wanted to move but could not.
        moves: Total moves made (energy used).
        mode: How the last decision was made (for telemetry / dashboard).
    """

    def __init__(self, agent_id: int, start: Cell, grid_size: int, battery: int) -> None:
        """Create a robot at ``start`` with empty knowledge and full battery."""
        self.agent_id: int = agent_id
        self.pos: Cell = start
        self.battery: int = battery
        self.alive: bool = True
        self.visits: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.visits[start] = 1
        self.known: np.ndarray = np.full((grid_size, grid_size), UNKNOWN, dtype=np.int8)
        self.known[start] = FREE
        self.stuck_ticks: int = 0
        self.moves: int = 0
        self.mode: str = MODE_IDLE

    @property
    def active(self) -> bool:
        """True if the robot can still move (alive with battery left)."""
        return self.alive and self.battery > 0

    def fail(self) -> None:
        """Permanently disable the robot (it stays in place as an obstacle)."""
        self.alive = False
        self.mode = MODE_OFF

    def sense(self, grid: np.ndarray, sense_range: int) -> int:
        """Scan the ground truth within a square window of ``sense_range``.

        Args:
            grid: True occupancy grid.
            sense_range: Chebyshev sensing radius.

        Returns:
            Number of walls discovered where the robot believed free space
            (i.e. debris that fell on a previously known route).
        """
        if not self.alive:
            return 0
        size = grid.shape[0]
        r, c = self.pos
        r0, r1 = max(0, r - sense_range), min(size, r + sense_range + 1)
        c0, c1 = max(0, c - sense_range), min(size, c + sense_range + 1)
        before = self.known[r0:r1, c0:c1]
        truth = grid[r0:r1, c0:c1]
        surprises = int(((before == FREE) & (truth == WALL)).sum())
        self.known[r0:r1, c0:c1] = truth
        return surprises

    def score(self, cell: Cell, others: list[Cell], cfg: SwarmConfig, rng: np.random.Generator) -> float:
        """Ant-pheromone move score (lower is better).

        ``score = W_p * visits + W_s * crowding + R * U(0, 1)``
        """
        return (
            cfg.pheromone_weight * float(self.visits[cell])
            + cfg.spread_weight * crowding(cell, others)
            + cfg.randomness * float(rng.random())
        )

    def choose_move(
        self,
        cfg: SwarmConfig,
        blocked: set[Cell],
        others: list[Cell],
        rng: np.random.Generator,
    ) -> Cell | None:
        """Decide this tick's move using only local knowledge.

        Args:
            cfg: Mission configuration (behaviour weights).
            blocked: Cells the robot perceives as occupied or reserved.
            others: Positions of perceived (alive) teammates for crowding.
            rng: Seeded random generator.

        Returns:
            The neighbouring cell to move into, or ``None`` to stay put.
        """
        if not self.active:
            self.mode = MODE_OFF
            return None
        size = self.known.shape[0]
        free_nbs = [nb for nb in neighbors4(self.pos, size) if self.known[nb] == FREE]
        options = [nb for nb in free_nbs if nb not in blocked]

        if self.stuck_ticks >= DEADLOCK_TICKS and options:
            self.mode = MODE_DEADLOCK
            return options[int(rng.integers(len(options)))]

        if any(self.visits[nb] == 0 for nb in free_nbs):
            if not options:
                self.mode = MODE_BLOCKED
                return None
            self.mode = MODE_PHEROMONE
            scores = [self.score(nb, others, cfg, rng) for nb in options]
            return options[int(np.argmin(scores))]

        # All neighbours visited: head for the nearest unvisited known-free cell.
        passable = self.known == FREE
        goals = passable & (self.visits == 0)
        if not goals.any():
            self.mode = MODE_IDLE  # nothing left to explore in my belief
            return None
        path = bfs_path(passable, self.pos, goals, blocked)
        if path is not None:
            self.mode = MODE_BFS
            return path[1]
        self.mode = MODE_BLOCKED  # a teammate is in the way; wait
        return None

    def commit(self, move: Cell | None) -> None:
        """Apply the outcome of this tick's decision.

        Args:
            move: Cell moved into, or ``None`` if the robot stayed.
        """
        if move is None:
            if self.mode == MODE_BLOCKED:
                self.stuck_ticks += 1
            elif self.mode in (MODE_IDLE, MODE_OFF):
                self.stuck_ticks = 0
            return
        self.pos = move
        self.battery -= 1
        self.moves += 1
        self.visits[move] += 1
        self.stuck_ticks = 0
