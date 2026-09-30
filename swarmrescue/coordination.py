"""Decentralized coordination: the robots' radio link and local perception.

There is no central controller and so no single point of failure. Robots
only exchange pheromone trails and debris maps with teammates inside radio
range, and they only perceive nearby robots.
"""

from __future__ import annotations

import numpy as np

from swarmrescue.agent import Robot
from swarmrescue.world import Cell, manhattan

# Once the radio link fails, robots can only detect others with proximity sensors.
PROXIMITY_RANGE: int = 2


class RadioLink:
    """Short-range radio: peer-to-peer links between robots, no base station (no single point of failure).

    Attributes:
        comm_range: Manhattan radio range.
        online: False after the aftershock cuts the radio link.
    """

    def __init__(self, comm_range: int) -> None:
        """Create a working radio link with the given range."""
        self.comm_range: int = comm_range
        self.online: bool = True

    def cut(self) -> None:
        """Take the radio link down (Round 2 aftershock)."""
        self.online = False

    @property
    def perception_radius(self) -> int:
        """Radius within which a robot can locate teammates.

        This is the radio range while the link is up, and otherwise the short
        proximity-sensor range. It is always at least 2, so adjacent robots
        are always seen and movement stays collision-free.
        """
        return max(self.comm_range, PROXIMITY_RANGE) if self.online else PROXIMITY_RANGE

    def share_pheromone_trails(self, robots: list[Robot]) -> int:
        """Merge pheromone trails and debris maps between robots in range.

        Merging is single-hop and order-independent. Every robot combines its
        maps with a snapshot of each in-range teammate's maps using an
        element-wise maximum: debris beats free, free beats unknown, and the
        stronger pheromone wins. Failed robots do not transmit.

        Args:
            robots: The swarm.

        Returns:
            Number of undirected radio links used this tick (0 when offline).
        """
        if not self.online:
            return 0
        alive = [r for r in robots if r.alive]
        snapshot = {r.robot_id: (r.trail.copy(), r.known.copy()) for r in alive}
        links = 0
        for i, a in enumerate(alive):
            for b in alive[i + 1 :]:
                if manhattan(a.pos, b.pos) > self.comm_range:
                    continue
                links += 1
                a.absorb(*snapshot[b.robot_id])
                b.absorb(*snapshot[a.robot_id])
        return links


def perceive_teammates(
    robot: Robot, robots: list[Robot], radius: int, positions: np.ndarray | None = None
) -> tuple[set[Cell], list[Cell]]:
    """What ``robot`` can locally perceive about its teammates.

    Args:
        robot: The observing robot.
        robots: The whole swarm (only nearby members are returned).
        radius: Manhattan perception radius.
        positions: Optional ``(R, 2)`` array of current robot positions, kept
            up to date by the caller, so distances are one vectorised operation.

    Returns:
        ``(blocked, others)``. ``blocked`` holds the cells occupied by any
        perceived robot, including failed ones, which are physical obstacles;
        avoiding them keeps movement collision-free. ``others`` holds the
        positions of perceived *working* robots, used for the crowding term.
    """
    pos = np.asarray([r.pos for r in robots]) if positions is None else positions
    near = np.flatnonzero(np.abs(pos - np.asarray(robot.pos)).sum(axis=1) <= radius)
    blocked: set[Cell] = set()
    others: list[Cell] = []
    for i in near:
        other = robots[int(i)]
        if other is robot:
            continue
        blocked.add(other.pos)
        if other.alive:
            others.append(other.pos)
    return blocked, others
