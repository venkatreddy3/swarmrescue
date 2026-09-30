"""SwarmRescue - Rescue Mission Control (Streamlit dashboard).

Run with:  streamlit run app.py

Accessibility: every map element is encoded by shape *and* a colour-blind-safe
Okabe-Ito colour, explained in a text legend, and summarised in plain text.
"""

from __future__ import annotations

import time
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from swarmrescue.advisor import PRIORITIES, advise  # noqa: E402
from swarmrescue.config import SwarmConfig  # noqa: E402
from swarmrescue.optimizer import PSOResult, run_pso  # noqa: E402
from swarmrescue.settings import SEED_BOUNDS, load_settings  # noqa: E402
from swarmrescue.simulation import Frame, SimulationResult, simulate  # noqa: E402

# Okabe-Ito colour-blind-safe palette.
C_UNEXPLORED = "#F2F2F2"
C_EXPLORED = "#56B4E9"   # sky blue
C_WALL = "#222222"       # near black
C_NEW_DEBRIS = "#8C8C8C"  # grey + "x" marker
C_ROBOT = "#E69F00"      # orange circle
C_FAILED = "#D55E00"     # vermillion X
C_FOUND = "#009E73"      # bluish green star
C_MISSING = "#CC79A7"    # reddish purple triangle

SEVERITY_LABEL = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO", "success": "OK"}
WEIGHT_DEFAULTS = {"pheromone_weight": 1.0, "spread_weight": 0.5, "randomness": 0.1, "evaporation_rate": 0.01}


@st.cache_data(show_spinner=False)
def run_mission(cfg_dict: dict[str, Any], round_no: int) -> SimulationResult:
    """Simulate one mission (cached on the configuration)."""
    return simulate(SwarmConfig(**cfg_dict), round_no=round_no, record_frames=True)


@st.cache_data(show_spinner=False)
def tune(cfg_dict: dict[str, Any], round_no: int, particles: int, iters: int) -> PSOResult:
    """Run PSO over seeds 1, 2, 3 (cached on the inputs)."""
    return run_pso(SwarmConfig(**cfg_dict), round_no, (1, 2, 3), particles, iters)


def render_frame(frame: Frame, survivors: tuple[tuple[int, int], ...], initial_grid: np.ndarray) -> Figure:
    """Draw one mission frame with shape- and colour-coded markers.

    Args:
        frame: Snapshot to draw.
        survivors: Survivor cells.
        initial_grid: Grid at tick 0 (to tell new debris apart).

    Returns:
        A matplotlib figure.
    """
    n = frame.grid.shape[0]
    layer = np.zeros((n, n), dtype=int)  # 0 unexplored
    layer[frame.visited] = 1
    layer[frame.grid == 1] = 2
    new_debris = (frame.grid == 1) & (initial_grid == 0)
    layer[new_debris] = 3
    cmap = ListedColormap([C_UNEXPLORED, C_EXPLORED, C_WALL, C_NEW_DEBRIS])

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=100)
    ax.imshow(layer, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.set_xticks(range(0, n, 5))
    ax.set_yticks(range(0, n, 5))
    ax.set_xlabel("column", fontsize=8)
    ax.set_ylabel("row", fontsize=8)
    ax.tick_params(which="both", length=0, labelsize=7)

    if new_debris.any():
        rr, cc = np.nonzero(new_debris)
        ax.scatter(cc, rr, marker="x", c="white", s=18, linewidths=1)
    for (r, c), found in zip(survivors, frame.found):
        if found:
            ax.scatter(c, r, marker="*", s=240, c=C_FOUND, edgecolors="black", linewidths=0.8, zorder=3)
        else:
            ax.scatter(c, r, marker="^", s=130, c=C_MISSING, edgecolors="black", linewidths=0.8, zorder=3)
    for rid, ((r, c), alive) in enumerate(zip(frame.positions, frame.alive)):
        if alive:
            ax.scatter(c, r, marker="o", s=230, c=C_ROBOT, edgecolors="black", linewidths=1, zorder=4)
            ax.text(c, r, str(rid), ha="center", va="center", fontsize=8, fontweight="bold", zorder=5)
        else:
            ax.scatter(c, r, marker="X", s=240, c=C_FAILED, edgecolors="black", linewidths=1, zorder=4)

    legend = [
        Patch(facecolor=C_UNEXPLORED, edgecolor="grey", label="Unexplored free cell"),
        Patch(facecolor=C_EXPLORED, label="Explored cell"),
        Patch(facecolor=C_WALL, label="Wall / debris"),
        Patch(facecolor=C_NEW_DEBRIS, edgecolor="white", hatch="xx", label="New debris (Round 2)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C_ROBOT, markeredgecolor="black",
               markersize=10, label="Robot (number = id)"),
        Line2D([], [], marker="X", color="none", markerfacecolor=C_FAILED, markeredgecolor="black",
               markersize=10, label="Failed robot"),
        Line2D([], [], marker="*", color="none", markerfacecolor=C_FOUND, markeredgecolor="black",
               markersize=13, label="Survivor found"),
        Line2D([], [], marker="^", color="none", markerfacecolor=C_MISSING, markeredgecolor="black",
               markersize=10, label="Survivor not yet found"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=8, frameon=False)
    ax.set_title(f"Tick {frame.tick} - coverage {frame.coverage:.0%}", fontsize=11)
    fig.tight_layout()
    return fig


def describe_frame(frame: Frame, survivors: tuple[tuple[int, int], ...]) -> str:
    """Plain-text description of a frame (for screen readers)."""
    robots = ", ".join(
        f"robot {i} at row {r} col {c}" + ("" if alive else " (FAILED)")
        for i, ((r, c), alive) in enumerate(zip(frame.positions, frame.alive))
    )
    return (
        f"Tick {frame.tick}: {frame.coverage:.0%} of reachable area explored; "
        f"{sum(frame.found)} of {len(survivors)} survivors found. {robots}."
    )


def show_map(result: SimulationResult) -> None:
    """Map view with tick slider and play button."""
    frames = result.frames
    initial = frames[0].grid
    last = len(frames) - 1
    col_a, col_b = st.columns([3, 1])
    tick = col_a.slider("Mission tick to display", 0, last, last, key="tick_slider")
    step = col_b.number_input("Animation step (ticks)", 1, 20, 5)
    play = col_b.button("Play animation", help="Replays the mission from tick 0")
    placeholder = st.empty()
    caption = st.empty()
    if play:
        for t in list(range(0, last, int(step))) + [last]:
            fig = render_frame(frames[t], result.survivors, initial)
            placeholder.pyplot(fig, clear_figure=True)
            plt.close(fig)
            caption.caption(describe_frame(frames[t], result.survivors))
            time.sleep(0.05)
    else:
        fig = render_frame(frames[tick], result.survivors, initial)
        placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)
        caption.caption(describe_frame(frames[tick], result.survivors))


def show_metrics(result: SimulationResult) -> None:
    """Headline mission metrics."""
    items = [
        ("Coverage of reachable area", f"{result.coverage:.1%}"),
        ("Survivors found", f"{result.survivors_found} / {result.survivors_total}"),
        ("Tick when 90% covered", "never" if result.tick_at_90 is None else str(result.tick_at_90)),
        ("Fitness score", f"{result.fitness:.2f}"),
        ("Energy used (moves)", str(result.energy_moves)),
        ("Collisions", str(result.collisions)),
        ("Max decision latency", f"{result.max_latency_ms:.2f} ms"),
        ("Debris found on known routes", str(result.wall_surprises)),
        ("Time to first survivor (ticks)", "none found" if result.first_survivor_tick is None else str(result.first_survivor_tick)),
        ("Ticks a robot heard a survivor ping", str(result.ping_detections)),
    ]
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for col, (label, value) in zip(cols, items[i:i + 2]):
            col.metric(label, value)


def apply_tuned() -> None:
    """Button callback: copy PSO's best weights into the weight sliders."""
    pso: PSOResult | None = st.session_state.get("pso")
    if pso is not None:
        for key, value in pso.best_params.items():
            st.session_state[key] = float(round(value, 3))


def sidebar() -> tuple[dict[str, Any], int, str]:
    """Operator inputs. Returns (config dict, round, priority)."""
    for key, value in WEIGHT_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    sb = st.sidebar
    sb.header("Mission settings")
    round_label = sb.radio(
        "Scenario", ["Round 1 - static building", "Round 2 - aftershock + failures"],
        help="Round 2: at tick 60 new debris falls, robot 0 fails and the radio link drops.",
    )
    round_no = 1 if round_label.startswith("Round 1") else 2
    agents = sb.slider("Number of robots", 1, 10, 4)
    grid = sb.slider("Building size (grid cells per side)", 10, 40, 20)
    density = sb.slider("Debris density", 0.0, 0.35, 0.18, 0.01)
    battery = sb.slider("Battery (moves per robot)", 20, 1000, 250, 10)
    survivors = sb.slider("Trapped survivors", 0, 15, 5)
    max_ticks = sb.slider("Mission time limit (ticks)", 50, 1000, 300, 10)
    seed = sb.number_input("Random seed (map layout)", SEED_BOUNDS[0], SEED_BOUNDS[1], load_settings().seed, step=1)
    priority = sb.selectbox("Operator priority (for advice)", PRIORITIES)
    with sb.expander("Behaviour weights (tunable by PSO)"):
        st.slider("Pheromone weight (avoid visited cells)", 0.0, 3.0, step=0.01, key="pheromone_weight")
        st.slider("Spread weight (avoid teammates)", 0.0, 3.0, step=0.01, key="spread_weight")
        st.slider("Randomness", 0.0, 1.0, step=0.01, key="randomness")
    with sb.expander("Adaptive features"):
        use_pings = st.checkbox("Survivor acoustic pings", value=True,
                                help="Survivors tap or signal; robots within range head straight to them.")
        ping_range = st.slider("Ping range (cells)", 1, 10, 4, disabled=not use_pings)
        use_evaporation = st.checkbox("Pheromone evaporation", value=False,
                                      help="Trails fade over time so long-searched areas are revisited.")
        st.slider("Evaporation rate per tick", 0.0, 0.2, step=0.005, key="evaporation_rate",
                  disabled=not use_evaporation)
    cfg = {
        "grid_size": grid, "num_agents": agents, "wall_density": density, "battery": battery,
        "num_survivors": survivors, "max_ticks": max_ticks, "seed": int(seed),
        "new_walls": min(25, grid * grid // 4),
        "pheromone_weight": st.session_state["pheromone_weight"],
        "spread_weight": st.session_state["spread_weight"],
        "randomness": st.session_state["randomness"],
        "use_pings": use_pings, "ping_range": ping_range,
        "use_evaporation": use_evaporation, "evaporation_rate": st.session_state["evaporation_rate"],
    }
    return cfg, round_no, priority


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="Rescue Mission Control", layout="wide")
    st.title("SwarmRescue - Rescue Mission Control")
    st.caption(
        "Decentralized ant-pheromone robot swarm searching a collapsed building. "
        "Each robot decides locally; no central controller."
    )
    cfg_dict, round_no, priority = sidebar()
    try:
        cfg = SwarmConfig(**cfg_dict)
    except ValueError as exc:
        st.error(f"Invalid settings: {exc}")
        return

    with st.spinner("Robots exploring..."):
        result = run_mission(cfg_dict, round_no)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Building map")
        show_map(result)
    with right:
        st.subheader("Mission metrics")
        show_metrics(result)
        st.subheader("Coverage over time")
        st.line_chart(pd.DataFrame({"coverage": result.coverage_curve}), x_label="tick", y_label="coverage")

    st.subheader("Weight tuning (Particle Swarm Optimisation)")
    c1, c2, c3 = st.columns([1, 1, 2])
    particles = c1.slider("Particles", 2, 12, 8)
    iters = c2.slider("Iterations", 1, 20, 12)
    c3.write("Fitness is averaged over maps 1, 2 and 3 with the current mission settings.")
    if st.button("Tune weights with PSO"):
        with st.spinner("Particles searching the weight space..."):
            st.session_state["pso"] = tune(cfg_dict, round_no, particles, iters)
    pso: PSOResult | None = st.session_state.get("pso")
    if pso is not None:
        hist = pd.DataFrame(
            {"best fitness": [h.best_fitness for h in pso.history],
             "swarm mean fitness": [h.mean_fitness for h in pso.history]},
            index=pd.Index([h.iteration for h in pso.history], name="iteration"),
        )
        st.line_chart(hist, x_label="PSO iteration", y_label="fitness")
        st.write(
            f"Best weights: {', '.join(f'{k} = {v:.3f}' for k, v in pso.best_params.items())} "
            f"(fitness {pso.baseline_fitness:.2f} -> {pso.best_fitness:.2f})"
        )
        with st.expander("Convergence log"):
            st.code("\n".join(h.format() for h in pso.history))
        st.button("Apply tuned weights", on_click=apply_tuned)

    st.subheader("Mission Advisor")
    for rec in advise(result, cfg, priority, pso):
        text = f"**{SEVERITY_LABEL[rec.severity]} - {rec.title}.** {rec.message}"
        if rec.changes:
            text += "  \nSuggested settings: " + ", ".join(f"`{k} = {v}`" for k, v in rec.changes.items())
        {"critical": st.error, "warning": st.warning, "info": st.info, "success": st.success}[rec.severity](text)


if __name__ == "__main__":
    main()
