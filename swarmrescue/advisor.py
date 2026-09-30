"""Rule-based Mission Advisor for the search-and-rescue operator.

Turns mission metrics (coverage, survivors found, energy and battery use,
collisions, decision latency) and operator preferences into plain-English
recommendations with concrete parameter changes. It is fully offline and
deterministic: no language model, no network, no API keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from swarmrescue.config import SwarmConfig
from swarmrescue.optimizer import PSOResult
from swarmrescue.simulation import SimulationResult

SEVERITY_ORDER: dict[str, int] = {"critical": 0, "warning": 1, "info": 2, "success": 3}
PRIORITIES: tuple[str, ...] = ("balanced", "coverage", "speed", "energy")
LATENCY_BUDGET_MS: float = 50.0


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
    if priority not in PRIORITIES:
        raise ValueError(f"priority must be one of {PRIORITIES}")
    runs = [results] if isinstance(results, SimulationResult) else list(results)
    if not runs:
        raise ValueError("at least one result is required")

    coverage = _mean([r.coverage for r in runs])
    survivors = _mean([r.survivors_ratio for r in runs])
    collisions = sum(r.collisions for r in runs)
    depleted = _mean([r.depleted_robots for r in runs])
    timed_out = any(r.ticks_run >= cfg.max_ticks for r in runs)
    latency = max(r.max_latency_ms for r in runs)
    energy_per_cell = _mean([r.energy_moves / max(1, r.visited_cells) for r in runs])
    t90 = [r.tick_at_90 for r in runs]
    round2 = any(r.round_no == 2 for r in runs)

    recs: list[Recommendation] = []
    add = recs.append

    if collisions:
        add(Recommendation(
            "critical", "Collisions detected",
            f"{collisions} collision(s) occurred. Halt the mission and inspect robot sensors before redeploying.",
        ))

    if coverage < 0.9:
        pct = f"{coverage:.0%}"
        if depleted > 0:
            add(Recommendation(
                "warning", "Batteries ran out",
                f"Coverage is only {pct} and robots stopped with empty batteries. Give them more battery.",
                {"battery": int(cfg.battery * 1.5)},
            ))
        elif timed_out:
            add(Recommendation(
                "warning", "Mission ran out of time",
                f"Coverage is only {pct} when the time limit hit. Add robots or allow more ticks.",
                {"num_agents": min(20, cfg.num_agents + 2), "max_ticks": int(cfg.max_ticks * 1.5)},
            ))
        else:
            add(Recommendation(
                "warning", "Areas left unexplored",
                f"Coverage is {pct} and robots believe they are done. Debris or a failed robot may be "
                "sealing off rooms without radio contact; send extra robots.",
                {"num_agents": min(20, cfg.num_agents + 1)},
            ))

    if survivors < 1.0:
        add(Recommendation(
            "warning", "Survivors missed",
            f"Only {survivors:.0%} of survivors were found. Spread the robots out more and add one robot.",
            {"spread_weight": round(min(3.0, cfg.spread_weight + 0.5), 2),
             "num_agents": min(20, cfg.num_agents + 1)},
        ))

    if any(t is None for t in t90):
        add(Recommendation(
            "warning", "90% coverage never reached",
            "The swarm never covered 90% of the reachable area, so it earns no speed bonus.",
            {"num_agents": min(20, cfg.num_agents + 1)},
        ))
    elif _mean([t for t in t90 if t is not None]) > 0.5 * cfg.max_ticks or priority == "speed":
        add(Recommendation(
            "info", "Speed up the search",
            f"90% coverage takes about {_mean([t for t in t90 if t is not None]):.0f} ticks. "
            "More robots and a stronger spread term reach it sooner.",
            {"num_agents": min(20, cfg.num_agents + 1),
             "spread_weight": round(min(3.0, cfg.spread_weight + 0.3), 2)},
        ))

    revisiting = energy_per_cell > 2.5 or (priority == "energy" and energy_per_cell > 1.5)
    if revisiting and round2:
        add(Recommendation(
            "info", "Robots are revisiting cells",
            f"About {energy_per_cell:.1f} moves per explored cell. Without radio, robots cannot share "
            "pheromone maps and re-search each other's areas; restoring map sharing helps most.",
        ))
    elif revisiting:
        add(Recommendation(
            "info", "Robots are revisiting cells",
            f"About {energy_per_cell:.1f} moves per explored cell. Lower randomness and raise the "
            "pheromone weight so robots avoid already-searched areas.",
            {"randomness": round(max(0.0, cfg.randomness - 0.05), 3),
             "pheromone_weight": round(min(3.0, cfg.pheromone_weight + 0.3), 2)},
        ))

    if priority == "coverage" and coverage < 1.0:
        add(Recommendation(
            "info", "Maximise coverage",
            "For full coverage, add battery headroom so no robot stops early.",
            {"battery": int(cfg.battery * 1.25)},
        ))

    if latency > LATENCY_BUDGET_MS:
        add(Recommendation(
            "warning", "Decision loop too slow",
            f"Peak decision latency was {latency:.1f} ms (> {LATENCY_BUDGET_MS:.0f} ms). "
            "Use a smaller grid or fewer robots per controller.",
        ))

    if cfg.wall_density >= 0.3:
        add(Recommendation(
            "info", "Heavily collapsed structure",
            f"Debris density {cfg.wall_density:.0%} fragments the building; some rooms may be unreachable.",
        ))

    if round2:
        add(Recommendation(
            "info", "Plan for failures",
            "Round 2 lost a robot and the radio link. Keep a spare robot in reserve and consider "
            "dropping radio relays so robots can keep sharing maps.",
        ))

    if pso is not None and pso.best_fitness > pso.baseline_fitness + 1e-9:
        add(Recommendation(
            "info", "Apply tuned weights",
            f"PSO improved mean fitness from {pso.baseline_fitness:.2f} to {pso.best_fitness:.2f}.",
            {k: round(v, 3) for k, v in pso.best_params.items()},
        ))

    if not any(r.severity in ("critical", "warning") for r in recs):
        add(Recommendation(
            "success", "Mission on track",
            f"{coverage:.0%} coverage, {survivors:.0%} of survivors found, zero collisions.",
        ))

    return sorted(recs, key=lambda r: SEVERITY_ORDER[r.severity])


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
        """Recommendations for these mission results (see :func:`advise`)."""
        return advise(results, cfg, self.priority, pso)
