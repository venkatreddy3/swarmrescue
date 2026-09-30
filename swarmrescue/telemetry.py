"""Mission telemetry: real-time budget, safety margins, deadlocks and recalibration time.

MissionControl feeds this recorder every tick. It turns the Track 05 hard
constraints into measured numbers:

* decision latency per tick against ``latency_budget_ms`` (a warning is logged
  the first time the budget is exceeded);
* minimum separation between robots and violations of ``safety_margin``;
* deadlock detections (the 3-tick breaker firing) and blocked robot-ticks;
* trajectory recalibration time: how long a robot needs to decide a new move
  in a tick where it has just sensed debris on a route it believed free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from swarmrescue.agent import MODE_BLOCKED, MODE_DEADLOCK
from swarmrescue.world import Cell

logger = logging.getLogger(__name__)
NO_NEIGHBOUR: int = 10**6


def min_pairwise_distance(positions: list[Cell]) -> int:
    """Smallest Manhattan distance between any two robots (vectorised, O(R^2))."""
    if len(positions) < 2:
        return NO_NEIGHBOUR
    pos = np.asarray(positions)
    dist = np.abs(pos[:, None, :] - pos[None, :, :]).sum(axis=2)
    np.fill_diagonal(dist, NO_NEIGHBOUR)
    return int(dist.min())


@dataclass
class MissionTelemetry:
    """Accumulates safety and real-time metrics for one mission."""

    latency_budget_ms: float
    safety_margin: int
    latencies: list[float] = field(default_factory=list)
    reroute_ms: list[float] = field(default_factory=list)
    budget_violations: int = 0
    min_separation: int = NO_NEIGHBOUR
    margin_violations: int = 0
    deadlocks_broken: int = 0
    blocked_ticks: int = 0

    def record_tick(self, elapsed_ms: float) -> None:
        """Latency of one tick of swarm decisions, checked against the real-time budget."""
        self.latencies.append(elapsed_ms)
        if elapsed_ms > self.latency_budget_ms:
            self.budget_violations += 1
            if self.budget_violations == 1:
                logger.warning("Tick took %.1f ms, over the %.0f ms budget", elapsed_ms, self.latency_budget_ms)

    def record_decision(self, mode: str, elapsed_ms: float, recalibrating: bool) -> None:
        """One robot's decision: deadlock/blocked counters and recalibration time."""
        if mode == MODE_DEADLOCK:
            self.deadlocks_broken += 1
        elif mode == MODE_BLOCKED:
            self.blocked_ticks += 1
        if recalibrating:
            self.reroute_ms.append(elapsed_ms)

    def record_positions(self, positions: list[Cell]) -> None:
        """Minimum robot separation after a tick, and safety-margin violations."""
        separation = min_pairwise_distance(positions)
        self.min_separation = min(self.min_separation, separation)
        if self.safety_margin and separation <= self.safety_margin:
            self.margin_violations += 1

    @property
    def max_reroute_ms(self) -> float:
        """Worst trajectory-recalibration time after sensing new debris (0 if none)."""
        return max(self.reroute_ms, default=0.0)
