"""Benchmarks: per-tick decision latency and stability under perturbation.

Usage:
    python scripts/benchmark.py              # full tables (about a minute)
    python scripts/benchmark.py --quick      # 1 seed, for CI / smoke tests

Latency is the wall-clock time of one tick of swarm decisions (sense, share,
perceive, decide, move) measured by MissionControl, across all robots.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmrescue.config import SwarmConfig
from swarmrescue.simulation import MissionControl, SimulationResult, evaluate

GRIDS: dict[int, dict[str, int]] = {
    20: {"max_ticks": 300, "battery": 250, "new_walls": 25},
    40: {"max_ticks": 600, "battery": 600, "new_walls": 100},
}
ROBOTS: tuple[int, ...] = (4, 8, 16)
PERTURBATIONS: dict[str, dict[str, object]] = {
    "baseline (default config)": {},
    "stronger aftershock (50 debris)": {"new_walls": 50},
    "short radio range (2)": {"comm_range": 2},
    "denser rubble (25% debris)": {"wall_density": 0.25},
    "noisy decisions (randomness 0.5)": {"randomness": 0.5},
    "weak batteries (150 moves)": {"battery": 150},
}


@dataclass(frozen=True)
class LatencyRow:
    """Latency statistics for one (grid, robots, round) setting."""

    grid: int
    robots: int
    round_no: int
    ticks: int
    mean_ms: float
    p95_ms: float
    max_ms: float
    bfs_calls: int
    cache_hits: int

    def markdown(self) -> str:
        """Markdown table row."""
        share = self.cache_hits / max(1, self.cache_hits + self.bfs_calls)
        return (
            f"| {self.grid}x{self.grid} | {self.robots} | {self.round_no} | {self.ticks} | {self.mean_ms:.2f} "
            f"| {self.p95_ms:.2f} | {self.max_ms:.2f} | {self.bfs_calls} | {share:.0%} |"
        )


def measure(grid: int, robots: int, round_no: int, seeds: list[int]) -> LatencyRow:
    """Run missions and pool their per-tick latencies."""
    latencies: list[float] = []
    bfs = hits = 0
    for seed in seeds:
        cfg = SwarmConfig().with_updates(grid_size=grid, num_agents=robots, seed=seed, **GRIDS[grid])
        mission = MissionControl(cfg, round_no)
        mission.run()
        latencies += mission.latencies
        bfs += sum(r.bfs_calls for r in mission.robots)
        hits += sum(r.route_cache_hits for r in mission.robots)
    arr = np.asarray(latencies)
    p95 = float(np.percentile(arr, 95))
    return LatencyRow(grid, robots, round_no, arr.size, float(arr.mean()), p95, float(arr.max()), bfs, hits)


def stability_row(name: str, changes: dict[str, object], round_no: int, seeds: list[int]) -> str:
    """Fitness mean, std and range under one perturbation (Markdown row)."""
    results: list[SimulationResult] = evaluate(SwarmConfig().with_updates(**changes), seeds, round_no)
    fit = np.asarray([r.fitness for r in results])
    cov = np.mean([r.coverage for r in results])
    surv = np.mean([r.survivors_ratio for r in results])
    col = sum(r.collisions for r in results)
    return (
        f"| {name} | {round_no} | {fit.mean():.2f} | {fit.std():.2f} | {fit.min():.2f}-{fit.max():.2f} "
        f"| {cov:.3f} | {surv:.0%} | {col} |"
    )


def main() -> None:
    """Print both benchmark tables as Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one seed per setting")
    quick = parser.parse_args().quick
    seeds = [1] if quick else [1, 2, 3]
    start = time.perf_counter()
    print("| Map | Robots | Round | Ticks | Mean ms | p95 ms | Max ms | BFS runs | Route-cache share |")
    print("|---|---|---|---|---|---|---|---|---|")
    for grid in GRIDS:
        for robots in ROBOTS:
            for round_no in (1, 2):
                print(measure(grid, robots, round_no, seeds).markdown(), flush=True)
    stab_seeds = [1, 2] if quick else list(range(1, 21))
    print(f"\nStability under perturbation (seeds 1..{stab_seeds[-1]})\n")
    print("| Perturbation | Round | Fitness mean | Std | Range | Coverage | Survivors | Collisions |")
    print("|---|---|---|---|---|---|---|---|")
    for name, changes in PERTURBATIONS.items():
        for round_no in (1, 2):
            print(stability_row(name, changes, round_no, stab_seeds), flush=True)
    print(f"\nTotal benchmark time: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
