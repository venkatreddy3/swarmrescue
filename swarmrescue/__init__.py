"""SwarmRescue: decentralized ant-pheromone multi-robot exploration.

A swarm of robots explores a collapsed building (occupancy grid) using only
local information, shared pheromone (visit-count) maps and BFS fallback
planning, with Particle Swarm Optimisation tuning the behaviour weights.
"""

from swarmrescue.config import SwarmConfig, validate_config

__all__ = ["SwarmConfig", "validate_config"]
__version__ = "1.0.0"
