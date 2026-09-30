"""Ablation study: effect of acoustic pings and pheromone evaporation.

Runs every variant on the same disaster zones (seeds 1..20 by default) for both
rounds and prints mean metrics as a Markdown table.

Usage:
    python scripts/ablation.py            # seeds 1..20
    python scripts/ablation.py --maps 5   # quicker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmrescue.config import SwarmConfig
from swarmrescue.simulation import SimulationResult, evaluate

VARIANTS: dict[str, dict[str, bool]] = {
    "Attempt 1 baseline (no pings, no evaporation)": {"use_pings": False, "use_evaporation": False},
    "+ evaporation only (rate 0.01)": {"use_pings": False, "use_evaporation": True},
    "+ acoustic pings only (range 4) - default": {"use_pings": True, "use_evaporation": False},
    "+ pings + evaporation (0.01)": {"use_pings": True, "use_evaporation": True},
}


TABLE_HEADER = (
    "| Variant | Coverage | Survivors | 1st survivor | All survivors found "
    "| 90% coverage | Energy | Collisions | Fitness |"
)


def all_found_tick(result: SimulationResult) -> int:
    """Tick at which the last survivor was found (mission length if some were missed)."""
    return max(t if t is not None else result.ticks_run for t in result.survivor_found_ticks)


def row(name: str, results: list[SimulationResult]) -> str:
    """One Markdown table row of mean metrics."""
    first = [r.first_survivor_tick for r in results if r.first_survivor_tick is not None]
    t90 = [r.tick_at_90 if r.tick_at_90 is not None else 300 for r in results]
    return (
        f"| {name} | {np.mean([r.coverage for r in results]):.3f} "
        f"| {np.mean([r.survivors_ratio for r in results]):.1%} "
        f"| {np.mean(first):.1f} | {np.mean([all_found_tick(r) for r in results]):.1f} "
        f"| {np.mean(t90):.1f} | {np.mean([r.energy_moves for r in results]):.0f} "
        f"| {sum(r.collisions for r in results)} | {np.mean([r.fitness for r in results]):.2f} |"
    )


def main() -> None:
    """Print the ablation tables for Round 1 and Round 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=int, default=20, choices=range(1, 101), metavar="1-100")
    seeds = list(range(1, parser.parse_args().maps + 1))
    for round_no in (1, 2):
        print(f"\nRound {round_no} - mean over seeds {seeds[0]}..{seeds[-1]}\n")
        print(TABLE_HEADER)
        print("|---|---|---|---|---|---|---|---|---|")
        for name, flags in VARIANTS.items():
            print(row(name, evaluate(SwarmConfig(**flags), seeds, round_no)))


if __name__ == "__main__":
    main()
