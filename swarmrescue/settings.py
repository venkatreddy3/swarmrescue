"""Optional runtime settings loaded safely from the environment or a ``.env`` file.

SwarmRescue needs no secrets or API keys. The few optional settings below
can come from environment variables or a local ``.env`` file (see
``.env.example``). Every value is parsed defensively:

* only known keys are read, and anything else in the file is ignored;
* values are type-checked and clamped to safe bounds;
* an invalid value falls back to the default with a warning and never crashes;
* the ``.env`` file is parsed as plain ``KEY=VALUE`` text and is never executed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
SEED_BOUNDS: tuple[int, int] = (0, 10_000)
MAX_ENV_FILE_BYTES: int = 64 * 1024
KNOWN_KEYS: tuple[str, ...] = ("SWARM_SEED", "LOG_LEVEL")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings.

    Attributes:
        seed: Base seed for default missions (``SWARM_SEED``, 0 to 10000, default 1).
        log_level: Logging verbosity (``LOG_LEVEL``, default ``WARNING``).
    """

    seed: int = 1
    log_level: str = "WARNING"

    @property
    def default_seeds(self) -> list[int]:
        """Three consecutive default seeds starting at ``seed``."""
        return [self.seed, self.seed + 1, self.seed + 2]


def parse_env_file(path: Path) -> dict[str, str]:
    """Read ``KEY=VALUE`` pairs for known keys from a ``.env`` file.

    Blank lines, comments, unknown keys and malformed lines are skipped.
    Files larger than 64 KB are ignored. Surrounding quotes are stripped.

    Args:
        path: Location of the ``.env`` file.

    Returns:
        A dictionary of the recognised keys found in the file.
    """
    if not path.is_file() or path.stat().st_size > MAX_ENV_FILE_BYTES:
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key in KNOWN_KEYS:
            values[key] = value.strip().strip("'\"")
    return values


def _parse_seed(raw: str | None, default: int) -> int:
    """Parse and bound ``SWARM_SEED``; fall back to ``default`` if invalid."""
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid SWARM_SEED=%r (not an integer)", raw[:20])
        return default
    low, high = SEED_BOUNDS
    if not low <= value <= high:
        logger.warning("SWARM_SEED=%d out of range [%d, %d]; clamping", value, low, high)
    return min(max(value, low), high)


def _parse_log_level(raw: str | None, default: str) -> str:
    """Parse ``LOG_LEVEL``; fall back to ``default`` if it is not a known level."""
    if raw is None or raw == "":
        return default
    level = raw.strip().upper()
    if level not in LOG_LEVELS:
        logger.warning("Ignoring invalid LOG_LEVEL=%r", raw[:20])
        return default
    return level


def load_settings(env: Mapping[str, str] | None = None, env_file: Path | None = Path(".env")) -> Settings:
    """Load settings with safe defaults.

    Precedence (highest first): process environment, then the ``.env`` file,
    then the built-in defaults.

    Args:
        env: Environment mapping (defaults to ``os.environ``).
        env_file: Optional ``.env`` path; ``None`` disables file loading.

    Returns:
        A validated :class:`Settings`.
    """
    merged: dict[str, str] = parse_env_file(env_file) if env_file is not None else {}
    source = os.environ if env is None else env
    merged.update({k: source[k] for k in KNOWN_KEYS if k in source})
    defaults = Settings()
    return Settings(
        seed=_parse_seed(merged.get("SWARM_SEED"), defaults.seed),
        log_level=_parse_log_level(merged.get("LOG_LEVEL"), defaults.log_level),
    )


def configure_logging(settings: Settings) -> None:
    """Apply the configured log level to the root logger."""
    logging.basicConfig(level=getattr(logging, settings.log_level), format="%(levelname)s %(name)s: %(message)s")
