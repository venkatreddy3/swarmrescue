"""SwarmRescue - Rescue Mission Control (Streamlit dashboard).

Run with:  streamlit run app.py

Accessibility: every map element is encoded by shape *and* a colour-blind-safe
Okabe-Ito colour, explained in a text legend, and summarised in plain text.
Every control has help text, and a "How to use this dashboard" guide is built in.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("swarmrescue.app")
ROOT = Path(__file__).resolve().parent
PACKAGE = "swarmrescue"
IMPORT_TIME_ATTR = "__imported_at__"


def _package_modules(package: str) -> dict[str, Any]:
    """Currently imported modules that belong to ``package``."""
    return {n: m for n, m in sys.modules.items() if n == package or n.startswith(package + ".")}


def _is_stale(modules: Mapping[str, Any], package: str, package_dir: Path) -> bool:
    """True if any module is from outside ``package_dir`` or older than its source file."""
    imported_at = getattr(modules.get(package), IMPORT_TIME_ATTR, None)
    if not isinstance(imported_at, float):
        return True  # a copy from before this safeguard existed
    for module in modules.values():
        file = getattr(module, "__file__", None)
        if not file or package_dir not in Path(file).resolve().parents:
            return True
        if Path(file).exists() and Path(file).stat().st_mtime > imported_at:
            return True
    return False


def ensure_fresh_package(root: Path = ROOT, package: str = PACKAGE) -> list[str]:
    """Make ``import swarmrescue`` load the repository's *current* code.

    Streamlit Cloud keeps one Python process alive across ``git pull`` updates
    and re-runs ``app.py``, but it does not always reload helper modules that
    were already imported. The new ``app.py`` then meets an old
    ``swarmrescue`` (``TypeError: unexpected keyword argument`` or
    ``ImportError: cannot import name``). This function:

    1. puts ``root`` first on ``sys.path`` so the local package always wins;
    2. drops the imported ``package`` modules if any is stale: loaded from
       another location, from a copy without an import timestamp, or with a
       source file modified after the package was imported.

    Returns:
        Names of the modules that were purged (empty when everything is fresh).
    """
    root_str = str(root)
    if not sys.path or sys.path[0] != root_str:
        sys.path[:] = [root_str] + [p for p in sys.path if p != root_str]
    loaded = _package_modules(package)
    if not loaded or not _is_stale(loaded, package, (root / package).resolve()):
        return []
    for name in loaded:
        del sys.modules[name]
    importlib.invalidate_caches()
    return sorted(loaded)


PURGED_MODULES = ensure_fresh_package()

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from swarmrescue.advisor import PRIORITIES, Recommendation, advise  # noqa: E402
from swarmrescue.config import SwarmConfig  # noqa: E402
from swarmrescue.optimizer import PSOResult, run_pso  # noqa: E402
from swarmrescue.render import describe_frame, render_frame  # noqa: E402
from swarmrescue.settings import SEED_BOUNDS, load_settings  # noqa: E402
from swarmrescue.simulation import REASON_COMPLETE, SimulationResult, simulate  # noqa: E402
from swarmrescue.ui_state import (  # noqa: E402
    WEIGHT_DEFAULTS,
    apply_tuned_weights,
    build_config_dict,
    sanitize_weights,
)

__all__ = [
    "WEIGHT_DEFAULTS",
    "apply_tuned_weights",
    "build_config_dict",
    "describe_frame",
    "ensure_fresh_package",
    "render_frame",
    "sanitize_weights",
]

CODE_VERSION = hashlib.sha256(
    repr(sorted((n, getattr(m, "__file__", "")) for n, m in _package_modules(PACKAGE).items())).encode()
    + repr(getattr(sys.modules[PACKAGE], IMPORT_TIME_ATTR, 0.0)).encode()
).hexdigest()[:12]
SEVERITY_LABEL = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO", "success": "OK"}
ROUND_LABELS = ("Round 1 - static building", "Round 2 - aftershock + failures")
HOW_TO = """
1. **Set the mission** in the sidebar: scenario, number of robots, building size, debris, battery and survivors.
2. **Watch the search** on the map: drag the tick slider or press *Play animation*. Robots are orange circles,
   survivors are stars (found) or triangles (not yet found), and a vermillion X is a failed robot.
3. **Read the end-of-mission message and metrics**: coverage, time to first survivor, energy and safety.
4. **Tune** (optional): press *Tune weights with PSO*, check the convergence chart, then *Apply tuned weights*.
5. **Follow the Mission Advisor**: plain-English recommendations, each with suggested settings.
"""


@st.cache_data(show_spinner=False, max_entries=64)
def run_mission(cfg_dict: dict[str, Any], round_no: int, code_version: str = "") -> SimulationResult:
    """Simulate one mission (cached on the configuration and the simulator's code version)."""
    return simulate(SwarmConfig(**cfg_dict), round_no=round_no, record_frames=True)


@st.cache_data(show_spinner=False, max_entries=16)
def tune(cfg_dict: dict[str, Any], round_no: int, particles: int, iters: int, code_version: str = "") -> PSOResult:
    """Run PSO over seeds 1, 2, 3 (cached on the inputs and the simulator's code version)."""
    return run_pso(SwarmConfig(**cfg_dict), round_no, (1, 2, 3), n_particles=particles, n_iters=iters)


def draw(result: SimulationResult, tick: int, placeholder: Any, caption: Any) -> None:
    """Render one frame of the map plus its screen-reader description."""
    frame = result.frames[tick]
    fig = render_frame(frame, result.survivors, result.frames[0].grid)
    placeholder.pyplot(fig, clear_figure=True)
    caption.caption(describe_frame(frame, result.survivors))


def show_map(result: SimulationResult) -> None:
    """Map view with a tick slider and a play button."""
    last = len(result.frames) - 1
    col_a, col_b = st.columns([3, 1])
    tick = col_a.slider(
        "Mission tick to display", 0, last, last, key="tick_slider", help="Drag to see the swarm at any earlier tick."
    )
    step = col_b.number_input("Animation step (ticks)", 1, 20, 5, help="Ticks skipped between animation frames.")
    play = col_b.button("Play animation", help="Replays the mission from tick 0")
    placeholder, caption = st.empty(), st.empty()
    if not play:
        draw(result, tick, placeholder, caption)
        return
    for t in [*range(0, last, int(step)), last]:
        draw(result, t, placeholder, caption)
        time.sleep(0.05)


def metric_items(result: SimulationResult) -> list[tuple[str, str]]:
    """Label and formatted value of every headline metric."""
    first = result.first_survivor_tick
    return [
        ("Coverage of reachable area", f"{result.coverage:.1%}"),
        ("Survivors found", f"{result.survivors_found} / {result.survivors_total}"),
        ("Tick when 90% covered", "never" if result.tick_at_90 is None else str(result.tick_at_90)),
        ("Fitness score", f"{result.fitness:.2f}"),
        ("Energy used (moves)", str(result.energy_moves)),
        ("Collisions", str(result.collisions)),
        ("Max decision latency", f"{result.max_latency_ms:.2f} ms"),
        ("Debris found on known routes", str(result.wall_surprises)),
        ("Time to first survivor (ticks)", "none found" if first is None else str(first)),
        ("Ticks a robot heard a survivor ping", str(result.ping_detections)),
        ("Minimum robot separation (cells)", str(result.min_separation)),
        ("Deadlocks detected and broken", str(result.deadlocks_broken)),
        ("Throughput (cells searched per tick)", f"{result.throughput:.2f}"),
        ("Max reroute time after new debris", f"{result.max_reroute_ms:.2f} ms"),
    ]


def show_metrics(result: SimulationResult) -> None:
    """End-of-mission message and headline metrics in a two-column grid."""
    (st.success if result.end_reason == REASON_COMPLETE else st.info)(result.end_message)
    items = metric_items(result)
    for i in range(0, len(items), 2):
        for col, (label, value) in zip(st.columns(2), items[i : i + 2], strict=False):
            col.metric(label, value)


def apply_tuned() -> None:
    """Button callback: copy PSO's best weights into the weight sliders."""
    pso: PSOResult | None = st.session_state.get("pso")
    if pso is not None:
        apply_tuned_weights(st.session_state, pso.best_params)  # type: ignore[arg-type]


def mission_inputs() -> dict[str, Any]:
    """Sidebar controls describing the disaster zone and the swarm (all bounded widgets)."""
    sb = st.sidebar
    grid = int(sb.slider("Building size (grid cells per side)", 10, 40, 20, help="Larger buildings take longer."))
    seed_lo, seed_hi = SEED_BOUNDS
    seed = sb.number_input(
        "Random seed (map layout)", seed_lo, seed_hi, load_settings().seed, step=1, help="Same seed, same building."
    )
    return {
        "num_agents": int(sb.slider("Number of robots", 1, 10, 4, help="Rescue robots deployed at the entry point.")),
        "grid_size": grid,
        "wall_density": float(sb.slider("Debris density", 0.0, 0.35, 0.18, 0.01, help="Fraction of blocked cells.")),
        "battery": int(sb.slider("Battery (moves per robot)", 20, 1000, 250, 10, help="Each move costs 1 unit.")),
        "num_survivors": int(sb.slider("Trapped survivors", 0, 15, 5, help="People hidden in reachable cells.")),
        "max_ticks": int(sb.slider("Mission time limit (ticks)", 50, 1000, 300, 10, help="The mission stops here.")),
        "seed": int(seed),
        "new_walls": min(25, grid * grid // 4),
    }


def weight_sliders() -> None:
    """Behaviour-weight sliders; their values live in session state (tunable by PSO)."""
    with st.sidebar.expander("Behaviour weights (tunable by PSO)"):
        st.slider(
            "Pheromone weight (avoid visited cells)", 0.0, 3.0, step=0.01, key="pheromone_weight",
            help="Higher: robots avoid cells that were already searched.",
        )  # fmt: skip
        st.slider(
            "Spread weight (avoid teammates)", 0.0, 3.0, step=0.01, key="spread_weight",
            help="Higher: robots fan out away from each other.",
        )  # fmt: skip
        st.slider("Randomness", 0.0, 1.0, step=0.01, key="randomness", help="Noise that breaks ties between moves.")


def feature_toggles() -> dict[str, Any]:
    """Adaptive-feature switches (acoustic pings, pheromone evaporation)."""
    with st.sidebar.expander("Adaptive features"):
        use_pings = st.checkbox(
            "Survivor acoustic pings", value=True, help="Survivors signal; robots within range head straight to them."
        )
        ping_range = st.slider(
            "Ping range (cells)", 1, 10, 4, disabled=not use_pings, help="How far away a robot hears a survivor."
        )
        use_evaporation = st.checkbox(
            "Pheromone evaporation", value=False, help="Trails fade over time so long-searched areas are revisited."
        )
        st.slider(
            "Evaporation rate per tick", 0.0, 0.2, step=0.005, key="evaporation_rate",
            disabled=not use_evaporation, help="Fraction of pheromone lost every tick.",
        )  # fmt: skip
        learning = st.checkbox(
            "Online learning",
            value=True,
            help="Each robot learns which weights search the most new cells per move.",
        )
        margin = st.slider(
            "Safety margin (cells)", 0, 3, 0, help="Clearance robots keep from each other when possible; 0 = off."
        )
    return {
        "use_pings": bool(use_pings),
        "ping_range": int(ping_range),
        "use_evaporation": bool(use_evaporation),
        "safety_margin": int(margin),
        "use_learning": bool(learning),
    }


def sidebar() -> tuple[dict[str, Any], list[str], int, str]:
    """Operator inputs. Returns (config kwargs, dropped keys, round, priority)."""
    repaired = sanitize_weights(st.session_state)  # type: ignore[arg-type]
    if repaired:
        logger.warning("Reset stale session values: %s", repaired)
    st.sidebar.header("Mission settings")
    round_label = st.sidebar.radio(
        "Scenario", ROUND_LABELS, help="Round 2: at tick 60 new debris falls, robot 0 fails and the radio drops."
    )
    inputs = mission_inputs()
    priority = st.sidebar.selectbox(
        "Operator priority (for advice)", PRIORITIES, help="What the Mission Advisor should optimise for."
    )
    weight_sliders()
    inputs |= feature_toggles()
    cfg, dropped = build_config_dict(inputs, st.session_state)  # type: ignore[arg-type]
    return cfg, dropped, 1 if round_label == ROUND_LABELS[0] else 2, str(priority)


def header() -> None:
    """Title, one-line summary and the "How to use" guide."""
    st.set_page_config(page_title="Rescue Mission Control", layout="wide")
    if PURGED_MODULES:
        logger.warning("Reloaded stale simulator modules: %s", PURGED_MODULES)
        st.cache_data.clear()
        st.session_state.pop("pso", None)
    st.title("SwarmRescue - Rescue Mission Control")
    st.caption("Decentralized ant-pheromone robot swarm searching a collapsed building. No central controller.")
    with st.expander("How to use this dashboard"):
        st.markdown(HOW_TO)


def validated_config(cfg_dict: dict[str, Any], dropped: list[str]) -> SwarmConfig | None:
    """Construct the SwarmConfig, reporting ignored or invalid settings to the operator."""
    st.session_state["last_config"] = dict(cfg_dict)
    if dropped:
        logger.warning("SwarmConfig on this server does not accept %s; is the package stale?", dropped)
        st.warning(
            "Some settings were ignored because the server is running an outdated copy of the "
            f"simulator ({', '.join(dropped)}). Reboot the app to load the latest version."
        )
    try:
        return SwarmConfig(**cfg_dict)
    except (ValueError, TypeError) as exc:
        st.error(f"Invalid settings: {exc}")
        return None


def show_mission(result: SimulationResult) -> None:
    """Map on the left; end message, metrics and coverage curve on the right."""
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Building map")
        show_map(result)
    with right:
        st.subheader("Mission metrics")
        show_metrics(result)
        st.subheader("Coverage over time")
        st.line_chart(pd.DataFrame({"coverage": result.coverage_curve}), x_label="tick", y_label="coverage")


def show_pso(pso: PSOResult) -> None:
    """Convergence chart, best weights, log and the Apply button."""
    hist = pd.DataFrame(
        {
            "best fitness": [h.best_fitness for h in pso.history],
            "swarm mean fitness": [h.mean_fitness for h in pso.history],
        },
        index=pd.Index([h.iteration for h in pso.history], name="iteration"),
    )
    st.line_chart(hist, x_label="PSO iteration", y_label="fitness")
    best = ", ".join(f"{k} = {v:.3f}" for k, v in pso.best_params.items())
    st.write(f"Best weights: {best} (fitness {pso.baseline_fitness:.2f} -> {pso.best_fitness:.2f})")
    with st.expander("Convergence log"):
        st.code("\n".join(h.format() for h in pso.history))
    st.button("Apply tuned weights", on_click=apply_tuned, help="Copy the tuned weights into the sidebar sliders.")


def tuning_section(cfg_dict: dict[str, Any], round_no: int) -> PSOResult | None:
    """PSO controls; returns the latest tuning result, if any."""
    st.subheader("Weight tuning (Particle Swarm Optimisation)")
    c1, c2, c3 = st.columns([1, 1, 2])
    particles = c1.slider("Particles", 2, 12, 8, help="Candidate weight sets explored in parallel.")
    iters = c2.slider("Iterations", 1, 20, 12, help="PSO update rounds (more = slower, usually better).")
    c3.write("Fitness is averaged over maps 1, 2 and 3 with the current mission settings.")
    if st.button("Tune weights with PSO", help="Runs PSO on the current settings (results are cached)."):
        with st.spinner("Particles searching the weight space..."):
            st.session_state["pso"] = tune(cfg_dict, round_no, particles, iters, CODE_VERSION)
    pso: PSOResult | None = st.session_state.get("pso")
    if pso is not None:
        show_pso(pso)
    return pso


def show_advice(recs: list[Recommendation]) -> None:
    """Mission Advisor output; severity is spelled out in text, not colour alone."""
    st.subheader("Mission Advisor")
    boxes = {"critical": st.error, "warning": st.warning, "info": st.info, "success": st.success}
    for rec in recs:
        text = f"**{SEVERITY_LABEL[rec.severity]} - {rec.title}.** {rec.message}"
        if rec.changes:
            text += "  \nSuggested settings: " + ", ".join(f"`{k} = {v}`" for k, v in rec.changes.items())
        boxes[rec.severity](text)


def main() -> None:
    """Render the dashboard."""
    header()
    cfg_dict, dropped, round_no, priority = sidebar()
    cfg = validated_config(cfg_dict, dropped)
    if cfg is None:
        return
    with st.spinner("Robots exploring..."):
        result = run_mission(cfg_dict, round_no, CODE_VERSION)
    show_mission(result)
    pso = tuning_section(cfg_dict, round_no)
    show_advice(advise(result, cfg, priority, pso))


if __name__ == "__main__":
    main()
