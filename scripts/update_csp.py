"""Regenerate the hash-pinned Content-Security-Policy in web/vercel.json.

Run after editing the inline <script> or <style> in web/index.html:

    python scripts/update_csp.py

tests/test_web.py fails if the hashes are out of date.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def csp_hash(block: str) -> str:
    """CSP ``'sha256-...'`` source expression for one inline block."""
    return "'sha256-" + base64.b64encode(hashlib.sha256(block.encode("utf-8")).digest()).decode() + "'"


def build_csp(html: str) -> str:
    """Strict CSP allowing only this page's own inline script and style."""
    script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    return (
        f"default-src 'none'; script-src {csp_hash(script)}; style-src {csp_hash(style)}; "
        "img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


def main() -> None:
    """Rewrite the Content-Security-Policy header value in web/vercel.json."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    path = WEB / "vercel.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for header in cfg["headers"][0]["headers"]:
        if header["key"] == "Content-Security-Policy":
            header["value"] = build_csp(html)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Updated", path)


if __name__ == "__main__":
    main()
