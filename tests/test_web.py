"""Tests for the static Vercel demo page (web/index.html + web/vercel.json)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.update_csp import csp_hash, inline_block

WEB = Path(__file__).resolve().parents[1] / "web"
VERCEL_URL = "https://swarmrescue-web.vercel.app/"
STREAMLIT_URL = "https://swarmrescue.streamlit.app/"
VIDEO_URL = "https://drive.google.com/drive/folders/163nfZIjv_LMQjkQQFXBVvoaKClOD8FxX?usp=drive_link"
HTML = (WEB / "index.html").read_text(encoding="utf-8")
SCRIPT = inline_block(HTML, "script")
STYLE = inline_block(HTML, "style")

# Headless harness: STUBS fake the handful of DOM APIs the page uses, the page's
# own script runs unchanged, then DRIVER presses Step / Aftershock / New disaster zone.
STUBS = r"""
const els = {};
function el(id) {
  if (!els[id]) els[id] = { id, value: id === "robots" ? "4" : "12", checked: true, disabled: false,
    textContent: "", attrs: {}, handlers: {}, width: 500, height: 500,
    addEventListener(t, f) { this.handlers[t] = f; }, setAttribute(k, v) { this.attrs[k] = v; },
    getContext() {
      return new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}),
                             set: (t, k, v) => { t[k] = v; return true; } });
    } };
  return els[id];
}
globalThis.document = { getElementById: el };
globalThis.window = { matchMedia: () => ({ matches: true }) };
globalThis.performance = { now: () => 0 };
"""
DRIVER = r"""
const out = [];
for (let map = 0; map < MAPS; map++) {
  if (map > 0) els.reset.handlers.click();
  let shocked = false;
  for (let t = 0; t < 300; t++) {
    if (!shocked && t === 60 && map % 2 === 1) { els.shock.handlers.click(); shocked = true; }
    els.step.handlers.click();
  }
  out.push({ collisions: +els["m-col"].textContent, coverage: parseFloat(els["m-cov"].textContent) / 100,
             survivors: els["m-surv"].textContent, radio: els["m-radio"].textContent, shocked,
             tick: +els["m-tick"].textContent, status: els.status.textContent });
}
console.log(JSON.stringify(out));
"""


def harness(maps: int) -> str:
    """Node program: DOM stubs + the page's own script + a driver over ``maps`` missions."""
    return "\n".join((f"const MAPS = {maps};", STUBS, SCRIPT, DRIVER))


def test_page_is_tiny_and_self_contained() -> None:
    """Single small file; no external scripts, styles, fonts, fetches or storage."""
    assert len(HTML.encode("utf-8")) < 40_000
    assert not re.search(r"<script[^>]+src=", HTML)
    assert not re.search(r"<link[^>]+rel=\"stylesheet\"", HTML)
    assert "@import" not in STYLE and "url(" not in STYLE
    for forbidden in ("fetch(", "XMLHttpRequest", "WebSocket", "localStorage", "document.cookie", "import("):
        assert forbidden not in SCRIPT, forbidden
    external = re.findall(r"(?:src|href)=\"(https?://[^\"]+)\"", HTML)
    assert all(
        url.startswith(("https://github.com/venkatreddy3/swarmrescue", STREAMLIT_URL, VIDEO_URL)) for url in external
    )
    assert not re.findall(r"src=\"https?://", HTML)  # links only; nothing is loaded from elsewhere


def test_page_metadata_and_semantics() -> None:
    """Viewport, title, description, language and landmark elements are present."""
    assert '<html lang="en">' in HTML
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in HTML
    assert re.search(r"<title>[^<]{10,}</title>", HTML)
    assert re.search(r'<meta name="description" content="[^"]{50,}">', HTML)
    for tag in ("<header>", "<main>", "<footer>", "<h1>", "<table>", "<caption"):
        assert tag in HTML


def test_page_accessibility_features() -> None:
    """Keyboard buttons, labelled controls, live region, canvas description and a text legend."""
    buttons = re.findall(r"<button[^>]*>", HTML)
    assert len(buttons) >= 4 and all('type="button"' in b for b in buttons)
    assert 'aria-pressed="false"' in HTML and 'aria-live="polite"' in HTML
    assert re.search(r'<canvas[^>]+role="img"[^>]+aria-label=', HTML)
    for control in ("robots", "speed", "pings"):
        assert f'for="{control}"' in HTML
    assert ":focus-visible" in STYLE and "prefers-color-scheme:dark" in STYLE
    assert "prefers-reduced-motion" in SCRIPT
    for colour, words in (
        ("#E69F00", "orange circle"),
        ("#009E73", "green star"),
        ("#CC79A7", "purple triangle"),
        ("#56B4E9", "sky blue"),
        ("#D55E00", "vermillion"),
    ):
        assert colour in HTML and words in HTML  # Okabe-Ito colour + shape/text, never colour alone


def test_page_shows_real_results_and_links() -> None:
    """Results table carries the README numbers; dashboard and repo links exist."""
    for value in ("126.95", "128.06", "116.94", "123.02"):
        assert value in HTML
    assert f'<a id="dashboard-link" href="{STREAMLIT_URL}">' in HTML
    assert 'href="https://github.com/venkatreddy3/swarmrescue"' in HTML


def test_readme_has_real_demo_links() -> None:
    """README lists the deployed Vercel and Streamlit URLs and no leftover placeholders."""
    readme = (WEB.parent / "README.md").read_text(encoding="utf-8")
    assert VERCEL_URL in readme and STREAMLIT_URL in readme and VIDEO_URL in readme
    assert VIDEO_URL in HTML
    assert "YOUR-" not in readme and "replace with the deployed URL" not in readme


def test_vercel_config_has_matching_csp() -> None:
    """vercel.json sends a strict CSP whose hashes match the current inline code."""
    cfg = json.loads((WEB / "vercel.json").read_text(encoding="utf-8"))
    headers = {h["key"]: h["value"] for h in cfg["headers"][0]["headers"]}
    csp = headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert csp_hash(SCRIPT) in csp, "script changed: regenerate the CSP hash in web/vercel.json"
    assert csp_hash(STYLE) in csp, "style changed: regenerate the CSP hash in web/vercel.json"
    assert headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_js_port_is_collision_free_across_maps() -> None:
    """The in-browser port never collides and keeps coverage in [0, 1], with and without an aftershock."""
    code = harness(16)
    proc = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=120, check=True)
    runs = json.loads(proc.stdout)
    assert len(runs) == 16
    for run in runs:
        assert run["collisions"] == 0
        assert 0.0 <= run["coverage"] <= 1.0
        assert run["radio"] == ("lost" if run["shocked"] else "online")
    assert sum(r["coverage"] for r in runs) / len(runs) > 0.9  # the port actually explores


END_MESSAGE = re.compile(
    r"^Mission (complete|ended) at tick (\d+): (.+) \((\d+)% coverage, (\d) of 5 survivors found, (\d+) collisions\)\.$"
)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_js_port_reports_real_end_reason() -> None:
    """When a mission stops, the status line gives the real reason, tick and metrics."""
    code = harness(12)
    runs = json.loads(
        subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=120, check=True).stdout
    )
    reasons = set()
    for run in runs:
        m = END_MESSAGE.match(run["status"])
        assert m, run["status"]
        verb, tick, reason, pct, found, collisions = m.groups()
        assert int(tick) == run["tick"] and int(collisions) == run["collisions"] == 0
        assert run["survivors"].startswith(found)
        assert int(pct) == int(run["coverage"] * 100 + 1e-9) or int(pct) == int(run["coverage"] * 100) - 1
        assert (verb == "complete") == (reason == "every reachable cell searched and every survivor found")
        reasons.add(reason)
    assert reasons <= {
        "every reachable cell searched and every survivor found",
        "robot batteries depleted",
        "time limit of 300 ticks reached",
        "robots have nothing left to search in their maps (remaining cells are cut off or unknown)",
    } | {r for r in reasons if "working robots out of battery" in r}


def test_js_end_reason_wording_matches_python() -> None:
    """The web page and the Python simulator use the same end-reason wording."""
    from swarmrescue.simulation import REASON_COMPLETE

    assert f'const COMPLETE = "{REASON_COMPLETE}";' in SCRIPT
    for phrase in (
        "robot batteries depleted",
        "all robots have failed",
        "time limit of ",
        "nothing left to search in their maps",
        "survivors found, ",
    ):
        assert phrase in SCRIPT
