"""Accessible map rendering for the Mission Control dashboard (matplotlib, no Streamlit).

Every element is encoded by *shape* and by a colour from the colour-blind-safe
Okabe-Ito palette, explained in a text legend, and summarised in plain text
for screen readers by :func:`describe_frame`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from swarmrescue.simulation import Frame
from swarmrescue.world import Cell

# Okabe-Ito colour-blind-safe palette.
C_UNEXPLORED = "#F2F2F2"
C_EXPLORED = "#56B4E9"  # sky blue
C_WALL = "#222222"  # near black
C_NEW_DEBRIS = "#8C8C8C"  # grey + "x" marker
C_ROBOT = "#E69F00"  # orange circle
C_FAILED = "#D55E00"  # vermillion X
C_FOUND = "#009E73"  # bluish green star
C_MISSING = "#CC79A7"  # reddish purple triangle


def _cell_layers(frame: Frame, initial_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Category per cell (0 unexplored, 1 explored, 2 debris, 3 aftershock debris) and the new-debris mask."""
    layer = np.zeros(frame.grid.shape, dtype=int)
    layer[frame.visited] = 1
    layer[frame.grid == 1] = 2
    new_debris = (frame.grid == 1) & (initial_grid == 0)
    layer[new_debris] = 3
    return layer, new_debris


def _draw_grid(ax: Axes, layer: np.ndarray) -> None:
    """Cell colours, thin white grid lines and row/column axes."""
    n = layer.shape[0]
    ax.imshow(layer, cmap=ListedColormap([C_UNEXPLORED, C_EXPLORED, C_WALL, C_NEW_DEBRIS]), vmin=0, vmax=3)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.set_xticks(range(0, n, 5))
    ax.set_yticks(range(0, n, 5))
    ax.set_xlabel("column", fontsize=8)
    ax.set_ylabel("row", fontsize=8)
    ax.tick_params(which="both", length=0, labelsize=7)


def _draw_markers(ax: Axes, frame: Frame, survivors: tuple[Cell, ...], new_debris: np.ndarray) -> None:
    """Aftershock debris crosses, survivors (star/triangle) and robots (circle/X)."""
    if new_debris.any():
        rr, cc = np.nonzero(new_debris)
        ax.scatter(cc, rr, marker="x", c="white", s=18, linewidths=1)
    for (r, c), found in zip(survivors, frame.found, strict=True):
        marker, size, colour = ("*", 240, C_FOUND) if found else ("^", 130, C_MISSING)
        ax.scatter(c, r, marker=marker, s=size, c=colour, edgecolors="black", linewidths=0.8, zorder=3)
    for rid, ((r, c), alive) in enumerate(zip(frame.positions, frame.alive, strict=True)):
        if alive:
            ax.scatter(c, r, marker="o", s=230, c=C_ROBOT, edgecolors="black", linewidths=1, zorder=4)
            ax.text(c, r, str(rid), ha="center", va="center", fontsize=8, fontweight="bold", zorder=5)
        else:
            ax.scatter(c, r, marker="X", s=240, c=C_FAILED, edgecolors="black", linewidths=1, zorder=4)


def _marker(marker: str, colour: str, size: int, label: str) -> Line2D:
    """Legend entry for a scatter marker."""
    return Line2D(
        [],
        [],
        marker=marker,
        color="none",
        markerfacecolor=colour,
        markeredgecolor="black",
        markersize=size,
        label=label,
    )


def legend_handles() -> list[Artist]:
    """Text legend: every colour is paired with a shape and a label."""
    return [
        Patch(facecolor=C_UNEXPLORED, edgecolor="grey", label="Unexplored free cell"),
        Patch(facecolor=C_EXPLORED, label="Explored cell"),
        Patch(facecolor=C_WALL, label="Wall / debris"),
        Patch(facecolor=C_NEW_DEBRIS, edgecolor="white", hatch="xx", label="New debris (Round 2)"),
        _marker("o", C_ROBOT, 10, "Robot (number = id)"),
        _marker("X", C_FAILED, 10, "Failed robot"),
        _marker("*", C_FOUND, 13, "Survivor found"),
        _marker("^", C_MISSING, 10, "Survivor not yet found"),
    ]


def render_frame(frame: Frame, survivors: tuple[Cell, ...], initial_grid: np.ndarray) -> Figure:
    """Draw one mission frame with shape- and colour-coded markers.

    Args:
        frame: Snapshot to draw.
        survivors: Survivor cells.
        initial_grid: Grid at tick 0 (to tell aftershock debris apart).

    Returns:
        A matplotlib figure (the caller closes it).
    """
    layer, new_debris = _cell_layers(frame, initial_grid)
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=100)
    _draw_grid(ax, layer)
    _draw_markers(ax, frame, survivors, new_debris)
    ax.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False
    )
    ax.set_title(f"Tick {frame.tick} - coverage {frame.coverage:.0%}", fontsize=11)
    fig.tight_layout()
    return fig


def describe_frame(frame: Frame, survivors: tuple[Cell, ...]) -> str:
    """Plain-text description of a frame (for screen readers)."""
    robots = ", ".join(
        f"robot {i} at row {r} col {c}" + ("" if alive else " (FAILED)")
        for i, ((r, c), alive) in enumerate(zip(frame.positions, frame.alive, strict=True))
    )
    return (
        f"Tick {frame.tick}: {frame.coverage:.0%} of reachable area explored; "
        f"{sum(frame.found)} of {len(survivors)} survivors found. {robots}."
    )
