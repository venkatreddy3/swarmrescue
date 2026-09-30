"""SwarmRescue command-line interface.

Examples:
    python main.py --round 1
    python main.py --round 2 --seeds 1 2 3
    python main.py --round 1 --tune
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

import numpy as np

from swarmrescue.advisor import PRIORITIES, advise
from swarmrescue.config import SwarmConfig
from swarmrescue.optimizer import IterationLog, run_pso
from swarmrescue.simulation import SimulationResult, evaluate

HEADER = (
    f"{'seed':>5} | {'coverage':>8} | {'survivors':>9} | {'t@90%':>5} | "
    f"{'energy':>6} | {'collis.':>7} | {'max lat ms':>10} | {'fitness':>8}"
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    p = argparse.ArgumentParser(description="SwarmRescue: decentralized ant-pheromone swarm exploration")
    p.add_argument("--round", type=int, choices=(1, 2), default=1, help="1 = static building, 2 = scenario shift")
    p.add_argument("--tune", action="store_true", help="tune weights with PSO and compare before/after")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="seeds (maps) to run / tune on")
    p.add_argument("--eval-seeds", type=int, nargs="*", default=list(range(101, 111)),
                   help="held-out seeds for an unbiased before/after comparison when tuning")
    p.add_argument("--particles", type=int, default=8, help="PSO particles")
    p.add_argument("--iters", type=int, default=12, help="PSO iterations")
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--survivors", type=int, default=5)
    p.add_argument("--battery", type=int, default=250)
    p.add_argument("--wall-density", type=float, default=0.18)
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--priority", choices=PRIORITIES, default="balanced", help="operator goal for the advisor")
    return p


def config_from_args(args: argparse.Namespace) -> SwarmConfig:
    """Build a validated :class:`SwarmConfig` from CLI arguments."""
    return SwarmConfig(
        grid_size=args.grid_size, num_agents=args.agents, num_survivors=args.survivors,
        battery=args.battery, wall_density=args.wall_density, max_ticks=args.max_ticks,
    )


def format_row(r: SimulationResult) -> str:
    """One table row for a single mission."""
    t90 = "-" if r.tick_at_90 is None else str(r.tick_at_90)
    return (
        f"{r.seed:>5} | {r.coverage:>8.3f} | {r.survivors_found:>4}/{r.survivors_total:<4} | {t90:>5} | "
        f"{r.energy_moves:>6} | {r.collisions:>7} | {r.max_latency_ms:>10.3f} | {r.fitness:>8.3f}"
    )


def summarize(results: Sequence[SimulationResult]) -> dict[str, float]:
    """Average metrics over several missions."""
    t90 = [r.tick_at_90 for r in results if r.tick_at_90 is not None]
    return {
        "coverage": float(np.mean([r.coverage for r in results])),
        "survivors": float(np.mean([r.survivors_ratio for r in results])),
        "t90": float(np.mean(t90)) if t90 else float("nan"),
        "energy": float(np.mean([r.energy_moves for r in results])),
        "collisions": float(sum(r.collisions for r in results)),
        "latency": float(max(r.max_latency_ms for r in results)),
        "fitness": float(np.mean([r.fitness for r in results])),
    }


def print_table(title: str, results: Sequence[SimulationResult]) -> dict[str, float]:
    """Print per-seed rows plus the mean, returning the summary."""
    print(f"\n{title}")
    print(HEADER)
    print("-" * len(HEADER))
    for r in results:
        print(format_row(r))
    s = summarize(results)
    print("-" * len(HEADER))
    print(
        f"{'mean':>5} | {s['coverage']:>8.3f} | {s['survivors']:>8.0%}  | {s['t90']:>5.0f} | "
        f"{s['energy']:>6.0f} | {s['collisions']:>7.0f} | {s['latency']:>10.3f} | {s['fitness']:>8.3f}"
    )
    return s


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: run missions, optionally tune, print report and advice."""
    args = build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    print("=" * 78)
    print(f"SwarmRescue mission report - Round {args.round}")
    if args.round == 2:
        print(f"  Scenario shift at tick {cfg.shift_tick}: {cfg.new_walls} new debris cells, "
              "robot 0 fails, radio link lost")
    print(f"  grid {cfg.grid_size}x{cfg.grid_size}, debris {cfg.wall_density:.0%}, robots {cfg.num_agents}, "
          f"survivors {cfg.num_survivors}, battery {cfg.battery}, max ticks {cfg.max_ticks}")
    print(f"  weights: pheromone={cfg.pheromone_weight}, spread={cfg.spread_weight}, randomness={cfg.randomness}")
    print("=" * 78)

    base = evaluate(cfg, args.seeds, args.round)
    before = print_table(f"Default weights - seeds {args.seeds}", base)

    pso = None
    final_cfg, final_results = cfg, base
    if args.tune:
        print(f"\nPSO tuning ({args.particles} particles x {args.iters} iterations, "
              f"w=0.6, c1=c2=1.5, fitness averaged over seeds {args.seeds})")

        def log(entry: IterationLog) -> None:
            """Print one convergence-log line."""
            print(entry.format(), flush=True)

        start = time.perf_counter()
        pso = run_pso(cfg, args.round, args.seeds, args.particles, args.iters, callback=log)
        print(f"PSO finished in {time.perf_counter() - start:.1f} s")
        final_cfg = pso.best_config(cfg)
        final_results = evaluate(final_cfg, args.seeds, args.round)
        after = print_table(f"Tuned weights {pso.best_params} - seeds {args.seeds}", final_results)
        print(f"\nFitness on tuning seeds: {before['fitness']:.3f} -> {after['fitness']:.3f}")
        if args.eval_seeds:
            held_before = summarize(evaluate(cfg, args.eval_seeds, args.round))
            held_after = summarize(evaluate(final_cfg, args.eval_seeds, args.round))
            print(f"Held-out seeds {args.eval_seeds[0]}..{args.eval_seeds[-1]} "
                  f"({len(args.eval_seeds)} maps, never seen by PSO):")
            for key in ("coverage", "survivors", "t90", "energy", "collisions", "fitness"):
                print(f"  {key:>10}: {held_before[key]:9.3f} -> {held_after[key]:9.3f}")

    print("\nMission Advisor:")
    for rec in advise(final_results, final_cfg, args.priority, pso):
        print(f"  - {rec.format()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
