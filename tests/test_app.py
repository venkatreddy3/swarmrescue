"""Headless tests for the Streamlit dashboard (streamlit.testing.AppTest)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def page_text(at: AppTest) -> str:
    """Concatenate captions, markdown and alert text on the page."""
    parts = [c.value for c in at.caption] + [m.value for m in at.markdown]
    parts += [a.value for kind in (at.success, at.info, at.warning, at.error) for a in kind]
    return "\n".join(str(p) for p in parts)


def test_dashboard_round1_renders() -> None:
    """Default run renders metrics, map description and advisor output."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Coverage of reachable area", "Collisions", "Max decision latency"} <= labels
    assert "survivors found" in page_text(at)
    assert at.success or at.info or at.warning


def test_dashboard_round2_and_tuning() -> None:
    """Switching to Round 2 shows the failed robot; quick PSO shows a best-weights line."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Round 2 - aftershock + failures").run()
    assert not at.exception
    assert "(FAILED)" in page_text(at)
    sliders = {s.label: s for s in at.slider}
    sliders["Particles"].set_value(2)
    sliders["Iterations"].set_value(1)
    next(b for b in at.button if b.label == "Tune weights with PSO").click().run()
    assert not at.exception
    assert any("Best weights" in m.value for m in at.markdown)


# --- Config building (regression tests for the Streamlit Cloud TypeError) -------------

import dataclasses  # noqa: E402

import app  # noqa: E402
from swarmrescue.config import TUNABLE_BOUNDS, SwarmConfig  # noqa: E402

WIDGET_INPUTS = {
    "grid_size": 20,
    "num_agents": 4,
    "wall_density": 0.18,
    "battery": 250,
    "num_survivors": 5,
    "max_ticks": 300,
    "seed": 1,
    "new_walls": 25,
    "use_pings": True,
    "ping_range": 4,
    "use_evaporation": True,
}


@dataclasses.dataclass(frozen=True)
class OutdatedSwarmConfig:
    """Stand-in for the Attempt 1 SwarmConfig still loaded on a stale server."""

    grid_size: int = 20
    wall_density: float = 0.18
    num_agents: int = 4
    num_survivors: int = 5
    max_ticks: int = 300
    battery: int = 250
    pheromone_weight: float = 1.0
    spread_weight: float = 0.5
    randomness: float = 0.1
    new_walls: int = 25
    seed: int = 0


def test_build_config_matches_swarmconfig_fields() -> None:
    """The dashboard's kwargs construct a SwarmConfig with nothing dropped."""
    state: dict[str, Any] = {}
    app.sanitize_weights(state)
    cfg, dropped = app.build_config_dict(WIDGET_INPUTS, state)
    assert dropped == []
    built = SwarmConfig(**cfg)
    assert built.use_evaporation and built.evaporation_rate == 0.01 and built.pheromone_weight == 1.0


def test_outdated_config_class_no_longer_crashes() -> None:
    """Root cause: new keys vs an old SwarmConfig raised TypeError; now they are filtered out."""
    state: dict[str, Any] = {}
    app.sanitize_weights(state)
    full = {**WIDGET_INPUTS, **state}
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        OutdatedSwarmConfig(**full)
    cfg, dropped = app.build_config_dict(WIDGET_INPUTS, state, OutdatedSwarmConfig)
    assert OutdatedSwarmConfig(**cfg).pheromone_weight == 1.0
    assert dropped == ["evaporation_rate", "ping_range", "use_evaporation", "use_pings"]


def test_stale_session_values_are_repaired() -> None:
    """Strings, None, NaN, booleans and out-of-range values from old sessions are reset or clamped."""
    state = {
        "pheromone_weight": "1.7",
        "spread_weight": None,
        "randomness": float("nan"),
        "evaporation_rate": 9.0,
        "old_tuned_weight": 4.2,
    }
    repaired = app.sanitize_weights(state)
    assert state["pheromone_weight"] == 1.7 and state["spread_weight"] == 0.5
    assert state["randomness"] == 0.1 and state["evaporation_rate"] == TUNABLE_BOUNDS["evaporation_rate"][1]
    assert set(repaired) == {"pheromone_weight", "spread_weight", "randomness", "evaporation_rate"}
    cfg, _ = app.build_config_dict(WIDGET_INPUTS, state)
    assert "old_tuned_weight" not in cfg
    SwarmConfig(**cfg)


def test_apply_tuned_weights_only_copies_known_clamped_keys() -> None:
    """Tuned params are clamped, and unknown keys from any PSO result are ignored."""
    state: dict[str, Any] = {}
    app.sanitize_weights(state)
    app.apply_tuned_weights(state, {"pheromone_weight": 2.3456, "evaporation_rate": 0.7, "legacy_weight": 1.0})
    assert state["pheromone_weight"] == 2.346 and state["evaporation_rate"] == 0.2
    assert "legacy_weight" not in state
    cfg, dropped = app.build_config_dict(WIDGET_INPUTS, state)
    assert dropped == [] and SwarmConfig(**cfg).pheromone_weight == 2.346


@pytest.mark.parametrize("round_label", ["Round 1 - static building", "Round 2 - aftershock + failures"])
def test_dashboard_config_round_trip_with_tuned_weights(round_label: str) -> None:
    """End to end: each round, then tune + Apply tuned weights, then rerun. Never a TypeError."""
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["pheromone_weight"] = "stale-value"  # leftovers from an older app version
    at.session_state["old_tuned_weight"] = 3.0
    at.run()
    at.sidebar.radio[0].set_value(round_label).run()
    assert not at.exception
    SwarmConfig(**at.session_state["last_config"])
    sliders = {s.label: s for s in at.slider}
    sliders["Particles"].set_value(2)
    sliders["Iterations"].set_value(1)
    next(b for b in at.button if b.label == "Tune weights with PSO").click().run()
    next(b for b in at.button if b.label == "Apply tuned weights").click().run()
    assert not at.exception
    cfg = at.session_state["last_config"]
    tuned = at.session_state["pso"].best_params
    for key, value in tuned.items():
        assert cfg[key] == pytest.approx(round(value, 3))
    assert SwarmConfig(**cfg).use_pings
    assert not any("outdated" in w.value for w in at.warning)
