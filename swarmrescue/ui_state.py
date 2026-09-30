"""Operator-input handling for the dashboard: safe session state and config building.

Pure functions over plain mappings (no Streamlit import), so they are unit
tested directly. They protect the dashboard against stale or malformed values
that survive in a browser session across app updates.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, MutableMapping
from typing import Any

from swarmrescue.config import TUNABLE_BOUNDS, SwarmConfig

WEIGHT_DEFAULTS: dict[str, float] = {
    "pheromone_weight": 1.0,
    "spread_weight": 0.5,
    "randomness": 0.1,
    "evaporation_rate": 0.01,
}


def _clean_weight(raw: Any, default: float, low: float, high: float) -> float:
    """Coerce ``raw`` to a finite float in ``[low, high]``, or return ``default``."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if isinstance(raw, bool) or not math.isfinite(value):
        value = default
    return min(max(value, low), high)


def sanitize_weights(state: MutableMapping[str, Any]) -> list[str]:
    """Repair stale or invalid behaviour-weight values in session state.

    Session state survives reruns and app updates, so it can hold values from
    an older version: strings, ``None``, NaN or out-of-range numbers. Each
    weight is coerced to a finite float clamped to its PSO bounds, or reset to
    its default. This must run *before* the weight widgets are created.

    Returns:
        The keys that had to be repaired.
    """
    repaired: list[str] = []
    for key, default in WEIGHT_DEFAULTS.items():
        raw = state.get(key, default)
        value = _clean_weight(raw, default, *TUNABLE_BOUNDS[key])
        if key in state and raw != value:
            repaired.append(key)
        state[key] = value
    return repaired


def apply_tuned_weights(state: MutableMapping[str, Any], best_params: Mapping[str, float]) -> None:
    """Copy PSO's best weights into session state (known keys only, clamped to bounds)."""
    for key, value in best_params.items():
        if key in WEIGHT_DEFAULTS:
            low, high = TUNABLE_BOUNDS[key]
            state[key] = min(max(float(round(value, 3)), low), high)


def build_config_dict(
    inputs: Mapping[str, Any],
    state: Mapping[str, Any],
    config_cls: type = SwarmConfig,
) -> tuple[dict[str, Any], list[str]]:
    """Build the keyword arguments for ``config_cls`` exactly as the dashboard does.

    Widget inputs are merged with the behaviour weights held in session state.
    Only fields that ``config_cls`` accepts are kept. This guards against a
    server still running an older copy of ``swarmrescue.config``, which
    otherwise crashes with ``TypeError: unexpected keyword argument``.

    Returns:
        ``(config kwargs, dropped keys)``.
    """
    merged = {**inputs, **{key: state[key] for key in WEIGHT_DEFAULTS if key in state}}
    accepted = {f.name for f in dataclasses.fields(config_cls)}
    cfg = {k: v for k, v in merged.items() if k in accepted}
    return cfg, sorted(set(merged) - accepted)
