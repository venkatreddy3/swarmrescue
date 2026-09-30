"""Tests for the rule-based Mission Advisor."""

from __future__ import annotations

import dataclasses

import pytest

from swarmrescue.advisor import SEVERITY_ORDER, Recommendation, advise
from swarmrescue.config import SwarmConfig
from swarmrescue.optimizer import PSOResult
from swarmrescue.simulation import SimulationResult, simulate


@pytest.fixture(scope="module")
def good_result() -> SimulationResult:
    """A successful default Round 1 mission."""
    return simulate(SwarmConfig(seed=1))


def titles(recs: list[Recommendation]) -> set[str]:
    """Set of recommendation titles."""
    return {r.title for r in recs}


def test_successful_mission(good_result: SimulationResult, default_cfg: SwarmConfig) -> None:
    """A clean mission gets a success message and no warnings."""
    recs = advise(good_result, default_cfg)
    assert "Mission on track" in titles(recs)
    assert not any(r.severity in ("critical", "warning") for r in recs)


def test_low_coverage_with_depleted_batteries(good_result: SimulationResult, default_cfg: SwarmConfig) -> None:
    """Low coverage + empty batteries -> recommend 50% more battery."""
    bad = dataclasses.replace(good_result, coverage=0.5, depleted_robots=3)
    recs = advise(bad, default_cfg)
    rec = next(r for r in recs if r.title == "Batteries ran out")
    assert rec.changes == {"battery": 375}


def test_low_coverage_timeout_adds_robots(good_result: SimulationResult, default_cfg: SwarmConfig) -> None:
    """Low coverage at the time limit -> add robots / ticks."""
    bad = dataclasses.replace(good_result, coverage=0.6, depleted_robots=0, ticks_run=300)
    rec = next(r for r in advise(bad, default_cfg) if r.title == "Mission ran out of time")
    assert rec.changes["num_agents"] == 6


def test_missed_survivors_increase_spread(good_result: SimulationResult, default_cfg: SwarmConfig) -> None:
    """Missing survivors -> spread weight goes up by 0.5 (capped at 3)."""
    bad = dataclasses.replace(good_result, survivors_found=3)
    rec = next(r for r in advise(bad, default_cfg) if r.title == "Survivors missed")
    assert rec.changes["spread_weight"] == 1.0
    capped = next(r for r in advise(bad, default_cfg.with_updates(spread_weight=2.8)) if r.title == "Survivors missed")
    assert capped.changes["spread_weight"] == 3.0


def test_collisions_are_critical_and_first(good_result: SimulationResult, default_cfg: SwarmConfig) -> None:
    """Collisions produce a critical recommendation sorted first."""
    bad = dataclasses.replace(good_result, collisions=2, survivors_found=1)
    recs = advise(bad, default_cfg)
    assert recs[0].severity == "critical"
    ranks = [SEVERITY_ORDER[r.severity] for r in recs]
    assert ranks == sorted(ranks)


def test_round2_and_pso_advice(default_cfg: SwarmConfig) -> None:
    """Round 2 results mention failures; a better PSO result is recommended."""
    r2 = simulate(default_cfg.with_updates(seed=2), round_no=2)
    pso = PSOResult({"pheromone_weight": 1.2, "spread_weight": 0.9, "randomness": 0.05}, 125.0, 120.0, ())
    recs = advise([r2], default_cfg, pso=pso)
    assert "Plan for failures" in titles(recs)
    apply = next(r for r in recs if r.title == "Apply tuned weights")
    assert apply.changes["spread_weight"] == 0.9


def test_priorities_and_validation(good_result: SimulationResult, default_cfg: SwarmConfig) -> None:
    """Operator priority changes advice; unknown priority is rejected."""
    assert "Speed up the search" in titles(advise(good_result, default_cfg, priority="speed"))
    with pytest.raises(ValueError):
        advise(good_result, default_cfg, priority="fastest")
    with pytest.raises(ValueError):
        advise([], default_cfg)
    assert all(r.format().startswith("[") for r in advise(good_result, default_cfg))


def test_mission_advisor_class(good_result: SimulationResult, default_cfg: SwarmConfig) -> None:
    """MissionAdvisor gives the same advice as advise() and validates priority."""
    from swarmrescue.advisor import MissionAdvisor

    advisor = MissionAdvisor("speed")
    assert advisor.advise(good_result, default_cfg) == advise(good_result, default_cfg, "speed")
    with pytest.raises(ValueError):
        MissionAdvisor("panic")


def test_advisor_suggests_pings_when_disabled(default_cfg: SwarmConfig) -> None:
    """With pings off and survivors missed, the advisor recommends turning pings on."""
    cfg = default_cfg.with_updates(use_pings=False)
    result = dataclasses.replace(simulate(cfg.with_updates(seed=1)), survivors_found=3)
    rec = next(r for r in advise(result, cfg) if r.title == "Listen for survivors")
    assert rec.changes == {"use_pings": True}
    assert "Listen for survivors" not in titles(advise(result, default_cfg))
