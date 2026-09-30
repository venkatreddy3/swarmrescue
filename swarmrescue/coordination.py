"""Decentralized coordination: peer-to-peer perception and map sharing.

There is no central controller. Robots only exchange information with
teammates inside their communication radius, and only perceive nearby robots.
"""

from __future__ import annotations

import numpy as np

from swarmrescue.agent import Agent
from swarmrescue.world import Cell, manhattan

# Once communication fails, robots can only detect others with proximity sensors.
PROXIMITY_RANGE: int = 2


def perception_radius(comm_enabled: bool, comm_range: int) -> int:
    """Radius within which a robot can locate teammates.

    Args:
        comm_enabled: Whether the radio link is working.
        comm_range: Configured radio range.

    Returns:
        ``comm_range`` with radio, else the short proximity-sensor range.
        Always at least 2, so adjacent robots are always seen.
    """
    return max(comm_range, PROXIMITY_RANGE) if comm_enabled else PROXIMITY_RANGE


def perceive(agent: Agent, agents: list[Agent], radius: int) -> tuple[set[Cell], list[Cell]]:
    """What ``agent`` can locally perceive about its teammates.

    Args:
        agent: The observing robot.
        agents: The whole swarm (only nearby members are returned).
        radius: Manhattan perception radius.

    Returns:
        ``(blocked, others)``: cells occupied by any perceived robot
        (including failed ones, which are physical obstacles), and positions of
        perceived *working* robots used for the crowding term.
    """
    blocked: set[Cell] = set()
    others: list[Cell] = []
    for other in agents:
        if other is agent or manhattan(agent.pos, other.pos) > radius:
            continue
        blocked.add(other.pos)
        if other.alive:
            others.append(other.pos)
    return blocked, others


def share_maps(agents: list[Agent], comm_range: int) -> int:
    """Merge pheromone and wall maps between robots in radio range.

    Merging is single-hop and order-independent: every robot combines its own
    maps with a snapshot of each in-range teammate's maps via ``np.maximum``
    (walls dominate free, free dominates unknown; visit counts take the max).
    Failed robots do not communicate.

    Args:
        agents: The swarm.
        comm_range: Manhattan radio range.

    Returns:
        Number of undirected communication links used this tick.
    """
    alive = [a for a in agents if a.alive]
    snapshot = {a.agent_id: (a.visits.copy(), a.known.copy()) for a in alive}
    links = 0
    for i, a in enumerate(alive):
        for b in alive[i + 1:]:
            if manhattan(a.pos, b.pos) > comm_range:
                continue
            links += 1
            vb, kb = snapshot[b.agent_id]
            va, ka = snapshot[a.agent_id]
            np.maximum(a.visits, vb, out=a.visits)
            np.maximum(a.known, kb, out=a.known)
            np.maximum(b.visits, va, out=b.visits)
            np.maximum(b.known, ka, out=b.known)
    return links
