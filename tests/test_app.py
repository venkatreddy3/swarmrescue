"""Headless tests for the Streamlit dashboard (streamlit.testing.AppTest)."""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def page_text(at: "AppTest") -> str:
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
