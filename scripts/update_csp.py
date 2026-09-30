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


def inline_block(html: str, tag: str) -> str:
    """Content of the page's single inline ``<tag>`` block."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", html, re.S)
    if match is None:
        raise ValueError(f"no inline <{tag}> block found")
    return match.group(1)


def build_csp(html: str) -> str:
    """Strict CSP allowing only this page's own inline script and style."""
    script = inline_block(html, "script")
    style = inline_block(html, "style")
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
