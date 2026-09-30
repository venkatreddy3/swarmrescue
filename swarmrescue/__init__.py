"""SwarmRescue: decentralized ant-pheromone multi-robot search and rescue.

A swarm of robots explores a collapsed building (occupancy grid) using only
local information, shared pheromone trails and BFS rerouting, with Particle
Swarm Optimisation tuning the behaviour weights.
"""

import time as _time

from swarmrescue.config import SwarmConfig, validate_config

__all__ = ["SwarmConfig", "validate_config"]
__version__ = "3.0.0"
# Wall-clock time this package was imported. app.py compares it with the source
# files' modification times to detect a stale copy kept alive by a long-running
# server process after a redeploy.
__imported_at__: float = _time.time()
