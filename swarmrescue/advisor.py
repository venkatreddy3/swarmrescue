"""Rule-based Mission Advisor for the search-and-rescue operator.

Turns mission metrics (coverage, survivors found, energy and battery use,
collisions, decision latency) and operator preferences into plain-English
recommendations with concrete parameter changes. It is fully offline and
deterministic: no language model, no network, no API keys.

Design: :class:`MissionStats` summarises one or more missions, and each rule
is a small function ``MissionStats -> list[Recommendation]``. :data:`RULES`
lists them in the order their advice is shown within a severity level.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarmrescue.config import SwarmConfig
from swarmrescue.optimizer import PSOResult
from swarmrescue.simulation import SimulationResult

SEVERITY_ORDER: dict[str, int] = {"critical": 0, "warning": 1, "info": 2, "success": 3}
PRIORITIES: tuple[str, ...] = ("balanced", "coverage", "speed", "energy")
LATENCY_BUDGET_MS: float = 50.0
MAX_ROBOTS: int = 20


@dataclass(frozen=True)
class Recommendation:
    """One piece of advice for the mission operator.

    Attributes:
        severity: ``critical``, ``warning``, ``info`` or ``success``.
        title: Short headline.
        message: Plain-English explanation and suggested action.
        changes: Suggested configuration changes (field -> new value).
    """

    severity: str
    title: str
    message: str
    changes: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        """Single-line text rendering (severity is spelled out, not colour-coded)."""
        extra = ""
        if self.changes:
            extra = " -> try " + ", ".join(f"{k}={v}" for k, v in self.changes.items())
        return f"[{self.severity.upper()}] {self.title}: {self.message}{extra}"


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean of a non-empty sequence."""
    return float(np.mean(list(values)))


@dataclass(frozen=True)
class MissionStats:
    """Aggregated metrics of one or more missions, plus the operator context."""

    cfg: SwarmConfig
    priority: str
    pso: PSOResult | None
    coverage: float
    survivors: float
    collisions: int
    depleted: float
    timed_out: bool
    latency_ms: float
    energy_per_cell: float
    tick_at_90: tuple[int | None, ...]
    first_survivor: tuple[int, ...]
    round2: bool

    @classmethod
    def from_runs(
        cls, runs: Sequence[SimulationResult], cfg: SwarmConfig, priority: str, pso: PSOResult | None
    ) -> MissionStats:
        """Summarise ``runs`` (metrics are averaged; collisions are summed)."""
        return cls(
            cfg=cfg,
            priority=priority,
            pso=pso,
            coverage=_mean([r.coverage for r in runs]),
            survivors=_mean([r.survivors_ratio for r in runs]),
            collisions=sum(r.collisions for r in runs),
            depleted=_mean([r.depleted_robots for r in runs]),
            timed_out=any(r.ticks_run >= cfg.max_ticks for r in runs),
            latency_ms=max(r.max_latency_ms for r in runs),
            energy_per_cell=_mean([r.energy_moves / max(1, r.visited_cells) for r in runs]),
            tick_at_90=tuple(r.tick_at_90 for r in runs),
            first_survivor=tuple(r.first_survivor_tick for r in runs if r.first_survivor_tick is not None),
            round2=any(r.round_no == 2 for r in runs),
        )

    @property
    def more_robots(self) -> int:
        """One more robot, capped at the configuration limit."""
        return min(MAX_ROBOTS, self.cfg.num_agents + 1)


def rule_collisions(s: MissionStats) -> list[Recommendation]:
    """Any collision is a critical safety failure."""
    if not s.collisions:
        return []
    msg = f"{s.collisions} collision(s) occurred. Halt the mission and inspect robot sensors before redeploying."
    return [Recommendation("critical", "Collisions detected", msg)]


def rule_low_coverage(s: MissionStats) -> list[Recommendation]:
    """Explain coverage below 90% by its most likely cause."""
    if s.coverage >= 0.9:
        return []
    pct = f"{s.coverage:.0%}"
    if s.depleted > 0:
        msg = f"Coverage is only {pct} and robots stopped with empty batteries. Give them more battery."
        return [Recommendation("warning", "Batteries ran out", msg, {"battery": int(s.cfg.battery * 1.5)})]
    if s.timed_out:
        msg = f"Coverage is only {pct} when the time limit hit. Add robots or allow more ticks."
        changes = {"num_agents": min(MAX_ROBOTS, s.cfg.num_agents + 2), "max_ticks": int(s.cfg.max_ticks * 1.5)}
        return [Recommendation("warning", "Mission ran out of time", msg, changes)]
    msg = (
        f"Coverage is {pct} and robots believe they are done. Debris or a failed robot may be "
        "sealing off rooms without radio contact; send extra robots."
    )
    return [Recommendation("warning", "Areas left unexplored", msg, {"num_agents": s.more_robots})]


def rule_missed_survivors(s: MissionStats) -> list[Recommendation]:
    """Missing survivors call for a wider spread and one more robot."""
    if s.survivors >= 1.0:
        return []
    msg = f"Only {s.survivors:.0%} of survivors were found. Spread the robots out more and add one robot."
    changes = {"spread_weight": round(min(3.0, s.cfg.spread_weight + 0.5), 2), "num_agents": s.more_robots}
    return [Recommendation("warning", "Survivors missed", msg, changes)]


def rule_enable_pings(s: MissionStats) -> list[Recommendation]:
    """Suggest acoustic pings when survivors are missed or found late."""
    slow = bool(s.first_survivor) and _mean(s.first_survivor) > 0.1 * s.cfg.max_ticks
    if s.cfg.use_pings or not (s.survivors < 1.0 or slow):
        return []
    msg = (
        "Turn on acoustic pings so robots head straight for trapped survivors they can hear; "
        "in tests this cut the time to reach every survivor by about a third."
    )
    return [Recommendation("info", "Listen for survivors", msg, {"use_pings": True})]


def rule_speed(s: MissionStats) -> list[Recommendation]:
    """Flag a slow (or never reached) 90%-coverage milestone."""
    if any(t is None for t in s.tick_at_90):
        msg = "The swarm never covered 90% of the reachable area, so it earns no speed bonus."
        return [Recommendation("warning", "90% coverage never reached", msg, {"num_agents": s.more_robots})]
    t90 = _mean([t for t in s.tick_at_90 if t is not None])
    if t90 <= 0.5 * s.cfg.max_ticks and s.priority != "speed":
        return []
    msg = f"90% coverage takes about {t90:.0f} ticks. More robots and a stronger spread term reach it sooner."
    changes = {"num_agents": s.more_robots, "spread_weight": round(min(3.0, s.cfg.spread_weight + 0.3), 2)}
    return [Recommendation("info", "Speed up the search", msg, changes)]


def rule_revisits(s: MissionStats) -> list[Recommendation]:
    """Too many moves per explored cell means wasted energy."""
    if not (s.energy_per_cell > 2.5 or (s.priority == "energy" and s.energy_per_cell > 1.5)):
        return []
    head = f"About {s.energy_per_cell:.1f} moves per explored cell."
    if s.round2:
        msg = (
            f"{head} Without radio, robots cannot share pheromone maps and re-search each other's areas; "
            "restoring map sharing helps most."
        )
        return [Recommendation("info", "Robots are revisiting cells", msg)]
    msg = f"{head} Lower randomness and raise the pheromone weight so robots avoid already-searched areas."
    changes = {
        "randomness": round(max(0.0, s.cfg.randomness - 0.05), 3),
        "pheromone_weight": round(min(3.0, s.cfg.pheromone_weight + 0.3), 2),
    }
    return [Recommendation("info", "Robots are revisiting cells", msg, changes)]


def rule_coverage_priority(s: MissionStats) -> list[Recommendation]:
    """Operators who prioritise coverage get battery headroom."""
    if s.priority != "coverage" or s.coverage >= 1.0:
        return []
    msg = "For full coverage, add battery headroom so no robot stops early."
    return [Recommendation("info", "Maximise coverage", msg, {"battery": int(s.cfg.battery * 1.25)})]


def rule_latency(s: MissionStats) -> list[Recommendation]:
    """Warn when the decision loop exceeds its real-time budget."""
    budget = s.cfg.latency_budget_ms
    if s.latency_ms <= budget:
        return []
    msg = (
        f"Peak decision latency was {s.latency_ms:.1f} ms (> {budget:.0f} ms). "
        "Use a smaller grid or fewer robots per controller."
    )
    return [Recommendation("warning", "Decision loop too slow", msg)]


def rule_dense_debris(s: MissionStats) -> list[Recommendation]:
    """Heavily collapsed buildings may contain sealed rooms."""
    if s.cfg.wall_density < 0.3:
        return []
    msg = f"Debris density {s.cfg.wall_density:.0%} fragments the building; some rooms may be unreachable."
    return [Recommendation("info", "Heavily collapsed structure", msg)]


def rule_failures(s: MissionStats) -> list[Recommendation]:
    """Round 2 lessons: redundancy against a single point of failure."""
    if not s.round2:
        return []
    msg = (
        "Round 2 lost a robot and the radio link. Keep a spare robot in reserve and consider "
        "dropping radio relays so robots can keep sharing maps."
    )
    return [Recommendation("info", "Plan for failures", msg)]


def rule_tuned_weights(s: MissionStats) -> list[Recommendation]:
    """Recommend PSO weights only when they beat the baseline."""
    if s.pso is None or s.pso.best_fitness <= s.pso.baseline_fitness + 1e-9:
        return []
    msg = f"PSO improved mean fitness from {s.pso.baseline_fitness:.2f} to {s.pso.best_fitness:.2f}."
    return [Recommendation("info", "Apply tuned weights", msg, {k: round(v, 3) for k, v in s.pso.best_params.items()})]


Rule = Callable[[MissionStats], list[Recommendation]]
RULES: tuple[Rule, ...] = (
    rule_collisions,
    rule_low_coverage,
    rule_missed_survivors,
    rule_enable_pings,
    rule_speed,
    rule_revisits,
    rule_coverage_priority,
    rule_latency,
    rule_dense_debris,
    rule_failures,
    rule_tuned_weights,
)


def advise(
    results: SimulationResult | Sequence[SimulationResult],
    cfg: SwarmConfig,
    priority: str = "balanced",
    pso: PSOResult | None = None,
) -> list[Recommendation]:
    """Generate recommendations from mission results.

    Args:
        results: One result or several (e.g. one per seed); metrics are averaged.
        cfg: The configuration that produced the results.
        priority: Operator goal: ``balanced``, ``coverage``, ``speed`` or ``energy``.
        pso: Optional tuning result to recommend applying.

    Returns:
        Recommendations sorted from most to least severe.

    Raises:
        ValueError: If ``priority`` is unknown or ``results`` is empty.
    """
    return MissionAdvisor(priority).advise(results, cfg, pso)


class MissionAdvisor:
    """Expert-system advisor attached to Mission Control.

    Attributes:
        priority: Operator goal: ``balanced``, ``coverage``, ``speed`` or ``energy``.
    """

    def __init__(self, priority: str = "balanced") -> None:
        """Create an advisor for the given operator priority.

        Raises:
            ValueError: If ``priority`` is unknown.
        """
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        self.priority = priority

    def advise(
        self,
        results: SimulationResult | Sequence[SimulationResult],
        cfg: SwarmConfig,
        pso: PSOResult | None = None,
    ) -> list[Recommendation]:
        """Recommendations for these mission results, most severe first (see :func:`advise`).

        Raises:
            ValueError: If ``results`` is empty.
        """
        runs = [results] if isinstance(results, SimulationResult) else list(results)
        if not runs:
            raise ValueError("at least one result is required")
        stats = MissionStats.from_runs(runs, cfg, self.priority, pso)
        recs = [rec for rule in RULES for rec in rule(stats)]
        if not any(r.severity in ("critical", "warning") for r in recs):
            msg = f"{stats.coverage:.0%} coverage, {stats.survivors:.0%} of survivors found, zero collisions."
            recs.append(Recommendation("success", "Mission on track", msg))
        return sorted(recs, key=lambda r: SEVERITY_ORDER[r.severity])
