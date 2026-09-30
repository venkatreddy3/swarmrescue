"""Tests for the static Vercel demo page (web/index.html + web/vercel.json)."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
HTML = (WEB / "index.html").read_text(encoding="utf-8")
SCRIPT = re.search(r"<script>(.*?)</script>", HTML, re.S).group(1)
STYLE = re.search(r"<style>(.*?)</style>", HTML, re.S).group(1)

# Headless harness: stubs the handful of DOM APIs the page uses, then drives
# the page's own buttons (Step, Aftershock, New disaster zone) many times.
HARNESS = r"""
const els = {};
function el(id) {
  if (!els[id]) els[id] = { id, value: id === "robots" ? "4" : "12", checked: true, disabled: false,
    textContent: "", attrs: {}, handlers: {}, width: 500, height: 500,
    addEventListener(t, f) { this.handlers[t] = f; }, setAttribute(k, v) { this.attrs[k] = v; },
    getContext() { return new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; } }); } };
  return els[id];
}
globalThis.document = { getElementById: el };
globalThis.window = { matchMedia: () => ({ matches: true }) };
globalThis.performance = { now: () => 0 };
eval(SCRIPT);
const out = [];
for (let map = 0; map < MAPS; map++) {
  if (map > 0) els.reset.handlers.click();
  let shocked = false;
  for (let t = 0; t < 300; t++) {
    if (!shocked && t === 60 && map % 2 === 1) { els.shock.handlers.click(); shocked = true; }
    els.step.handlers.click();
  }
  out.push({ collisions: +els["m-col"].textContent, coverage: parseFloat(els["m-cov"].textContent) / 100,
             survivors: els["m-surv"].textContent, radio: els["m-radio"].textContent, shocked });
}
console.log(JSON.stringify(out));
"""


def csp_hash(block: str) -> str:
    """CSP source expression for an inline block."""
    return "'sha256-" + base64.b64encode(hashlib.sha256(block.encode("utf-8")).digest()).decode() + "'"


def test_page_is_tiny_and_self_contained() -> None:
    """Single small file; no external scripts, styles, fonts, fetches or storage."""
    assert len(HTML.encode("utf-8")) < 40_000
    assert not re.search(r"<script[^>]+src=", HTML)
    assert not re.search(r"<link[^>]+rel=\"stylesheet\"", HTML)
    assert "@import" not in STYLE and "url(" not in STYLE
    for forbidden in ("fetch(", "XMLHttpRequest", "WebSocket", "localStorage", "document.cookie", "import("):
        assert forbidden not in SCRIPT, forbidden
    external = re.findall(r"(?:src|href)=\"(https?://[^\"]+)\"", HTML)
    assert all(url.startswith("https://github.com/venkatreddy3/swarmrescue") for url in external)


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
    for colour, words in (("#E69F00", "orange circle"), ("#009E73", "green star"), ("#CC79A7", "purple triangle"),
                          ("#56B4E9", "sky blue"), ("#D55E00", "vermillion")):
        assert colour in HTML and words in HTML  # Okabe-Ito colour + shape/text, never colour alone


def test_page_shows_real_results_and_links() -> None:
    """Results table carries the README numbers; dashboard and repo links exist."""
    for value in ("126.95", "128.06", "116.94", "123.02"):
        assert value in HTML
    assert 'id="dashboard-link"' in HTML
    assert 'href="https://github.com/venkatreddy3/swarmrescue"' in HTML


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
    code = f"const SCRIPT = {json.dumps(SCRIPT)}; const MAPS = 16;\n{HARNESS}"
    proc = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=120, check=True)
    runs = json.loads(proc.stdout)
    assert len(runs) == 16
    for run in runs:
        assert run["collisions"] == 0
        assert 0.0 <= run["coverage"] <= 1.0
        assert run["radio"] == ("lost" if run["shocked"] else "online")
    assert sum(r["coverage"] for r in runs) / len(runs) > 0.9  # the port actually explores
