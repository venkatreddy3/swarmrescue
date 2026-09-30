"""Security tests: safe settings loading and no secrets in the repository."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from swarmrescue.settings import Settings, load_settings, parse_env_file

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI / Anthropic style keys
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),  # GitHub token
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),  # private keys
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]"),
]


def test_defaults_without_env(tmp_path: Path) -> None:
    """No environment and no file gives the safe defaults."""
    s = load_settings(env={}, env_file=tmp_path / "missing.env")
    assert s == Settings(seed=1, log_level="WARNING")
    assert s.default_seeds == [1, 2, 3]


def test_env_overrides_file(tmp_path: Path) -> None:
    """Process environment takes precedence over the .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("SWARM_SEED=7\nLOG_LEVEL=debug\n", encoding="utf-8")
    assert load_settings(env={}, env_file=env_file) == Settings(seed=7, log_level="DEBUG")
    assert load_settings(env={"SWARM_SEED": "9"}, env_file=env_file).seed == 9


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"SWARM_SEED": "abc"}, Settings()),
        ({"SWARM_SEED": "-50"}, Settings(seed=0)),
        ({"SWARM_SEED": "99999999"}, Settings(seed=10_000)),
        ({"LOG_LEVEL": "LOUD"}, Settings()),
        ({"LOG_LEVEL": "$(rm -rf /)"}, Settings()),
    ],
)
def test_invalid_values_fall_back_safely(env: dict[str, str], expected: Settings) -> None:
    """Garbage or hostile values never crash and never pass through unbounded."""
    assert load_settings(env=env, env_file=None) == expected


def test_env_file_parser_ignores_unknown_and_malformed(tmp_path: Path) -> None:
    """Only known keys are read; comments, junk and unknown keys are skipped."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nexport SWARM_SEED='12'\nOPENAI_API_KEY=should-not-be-read\ngarbage line\n\nLOG_LEVEL=\"INFO\"\n",
        encoding="utf-8",
    )
    assert parse_env_file(env_file) == {"SWARM_SEED": "12", "LOG_LEVEL": "INFO"}


def test_oversized_env_file_is_ignored(tmp_path: Path) -> None:
    """A huge .env file is not read at all."""
    env_file = tmp_path / ".env"
    env_file.write_text("SWARM_SEED=5\n" + "#" * 70_000, encoding="utf-8")
    assert parse_env_file(env_file) == {}


def test_env_example_documents_settings_without_secrets() -> None:
    """.env.example exists, documents both settings, and is not git-ignored."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "SWARM_SEED=" in text and "LOG_LEVEL=" in text
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore and "!.env.example" in gitignore


def test_no_secrets_in_tracked_files() -> None:
    """No tracked file contains anything that looks like a credential."""
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    for name in files:
        path = ROOT / name
        if path.suffix.lower() in {".png", ".jpg", ".ico"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            assert not pattern.search(text), f"possible secret in {name}"
