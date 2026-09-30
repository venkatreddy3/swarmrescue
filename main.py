"""SwarmRescue command-line interface.

Examples:
    python main.py --round 1
    python main.py --round 2 --seeds 1 2 3
    python main.py --round 1 --tune
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from typing import Callable, Sequence

import numpy as np

from swarmrescue.advisor import PRIORITIES, advise
from swarmrescue.config import SwarmConfig
from swarmrescue.optimizer import IterationLog, run_pso
from swarmrescue.settings import Settings, configure_logging, load_settings
from swarmrescue.simulation import SimulationResult, evaluate

logger = logging.getLogger("swarmrescue.cli")

HEADER = (
    f"{'seed':>5} | {'coverage':>8} | {'survivors':>9} | {'1st surv':>8} | {'t@90%':>5} | "
    f"{'energy':>6} | {'collis.':>7} | {'max lat ms':>10} | {'fitness':>8}"
)


MAX_SEEDS: int = 50
MAX_PARTICLES: int = 30
MAX_ITERS: int = 50


def bounded_int(low: int, high: int) -> Callable[[str], int]:
    """argparse type: an integer within ``[low, high]``."""

    def parse(text: str) -> int:
        """Parse and range-check one integer argument."""
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"{value} is outside [{low}, {high}]")
        return value

    return parse


def bounded_float(low: float, high: float) -> Callable[[str], float]:
    """argparse type: a finite float within ``[low, high]``."""

    def parse(text: str) -> float:
        """Parse and range-check one float argument."""
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
        if not math.isfinite(value) or not low <= value <= high:
            raise argparse.ArgumentTypeError(f"{text} is outside [{low}, {high}]")
        return value

    return parse


def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    """Create the CLI argument parser; every numeric input is range-checked."""
    settings = settings or Settings()
    seed = bounded_int(0, 10_000)
    p = argparse.ArgumentParser(description="SwarmRescue: decentralized ant-pheromone search and rescue swarm")
    p.add_argument("--round", type=int, choices=(1, 2), default=1, help="1 = static building, 2 = aftershock")
    p.add_argument("--tune", action="store_true", help="tune weights with PSO and compare before/after")
    p.add_argument("--seeds", type=seed, nargs="+", default=settings.default_seeds,
                   help=f"seeds (disaster zones) to run / tune on, at most {MAX_SEEDS} (default from SWARM_SEED)")
    p.add_argument("--eval-seeds", type=seed, nargs="*", default=list(range(101, 111)),
                   help="held-out seeds for an unbiased before/after comparison when tuning")
    p.add_argument("--particles", type=bounded_int(1, MAX_PARTICLES), default=8, help="PSO particles (1-30)")
    p.add_argument("--iters", type=bounded_int(0, MAX_ITERS), default=12, help="PSO iterations (0-50)")
    p.add_argument("--grid-size", type=bounded_int(5, 100), default=20)
    p.add_argument("--agents", type=bounded_int(1, 20), default=4, help="number of robots (1-20)")
    p.add_argument("--survivors", type=bounded_int(0, 50), default=5)
    p.add_argument("--battery", type=bounded_int(1, 100_000), default=250)
    p.add_argument("--wall-density", type=bounded_float(0.0, 0.45), default=0.18, help="debris density (0-0.45)")
    p.add_argument("--max-ticks", type=bounded_int(1, 5000), default=300)
    p.add_argument("--priority", choices=PRIORITIES, default="balanced", help="operator goal for the advisor")
    p.add_argument("--evaporation", action="store_true", help="enable pheromone evaporation (trails fade)")
    p.add_argument("--evaporation-rate", type=bounded_float(0.0, 0.2), default=0.01,
                   help="pheromone fraction lost per tick (0-0.2)")
    p.add_argument("--no-pings", action="store_true", help="disable survivor acoustic pings")
    p.add_argument("--ping-range", type=bounded_int(1, 10), default=4, help="distance at which robots hear survivors")
    return p


def config_from_args(args: argparse.Namespace) -> SwarmConfig:
    """Build a validated :class:`SwarmConfig` from CLI arguments."""
    return SwarmConfig(
        grid_size=args.grid_size, num_agents=args.agents, num_survivors=args.survivors,
        battery=args.battery, wall_density=args.wall_density, max_ticks=args.max_ticks,
        use_evaporation=args.evaporation, evaporation_rate=args.evaporation_rate,
        use_pings=not args.no_pings, ping_range=args.ping_range,
    )


def format_row(r: SimulationResult) -> str:
    """One table row for a single mission."""
    t90 = "-" if r.tick_at_90 is None else str(r.tick_at_90)
    first = "-" if r.first_survivor_tick is None else str(r.first_survivor_tick)
    return (
        f"{r.seed:>5} | {r.coverage:>8.3f} | {r.survivors_found:>4}/{r.survivors_total:<4} | {first:>8} | {t90:>5} | "
        f"{r.energy_moves:>6} | {r.collisions:>7} | {r.max_latency_ms:>10.3f} | {r.fitness:>8.3f}"
    )


def summarize(results: Sequence[SimulationResult]) -> dict[str, float]:
    """Average metrics over several missions."""
    t90 = [r.tick_at_90 for r in results if r.tick_at_90 is not None]
    first = [r.first_survivor_tick for r in results if r.first_survivor_tick is not None]
    return {
        "first_survivor": float(np.mean(first)) if first else float("nan"),
        "coverage": float(np.mean([r.coverage for r in results])),
        "survivors": float(np.mean([r.survivors_ratio for r in results])),
        "t90": float(np.mean(t90)) if t90 else float("nan"),
        "energy": float(np.mean([r.energy_moves for r in results])),
        "collisions": float(sum(r.collisions for r in results)),
        "latency": float(max(r.max_latency_ms for r in results)),
        "fitness": float(np.mean([r.fitness for r in results])),
    }


def generalizes(held_out_before: float, held_out_after: float) -> bool:
    """Over-fitting guard: tuned weights must not lower fitness on unseen maps."""
    return held_out_after >= held_out_before


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
        f"{'mean':>5} | {s['coverage']:>8.3f} | {s['survivors']:>8.0%}  | {s['first_survivor']:>8.1f} | {s['t90']:>5.0f} | "
        f"{s['energy']:>6.0f} | {s['collisions']:>7.0f} | {s['latency']:>10.3f} | {s['fitness']:>8.3f}"
    )
    for r in results:
        print(f"  seed {r.seed}: {r.end_message}")
    return s


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: run missions, optionally tune, print report and advice."""
    settings = load_settings()
    configure_logging(settings)
    parser = build_parser(settings)
    args = parser.parse_args(argv)
    if len(args.seeds) > MAX_SEEDS or len(args.eval_seeds) > MAX_SEEDS:
        parser.error(f"at most {MAX_SEEDS} seeds are allowed")
    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    logger.info("Running round %d on seeds %s", args.round, args.seeds)

    print("=" * 78)
    print(f"SwarmRescue mission report - Round {args.round}")
    if args.round == 2:
        print(f"  Aftershock at tick {cfg.shift_tick}: {cfg.new_walls} new debris cells, "
              "robot 0 fails, radio link lost")
    print(f"  grid {cfg.grid_size}x{cfg.grid_size}, debris {cfg.wall_density:.0%}, robots {cfg.num_agents}, "
          f"survivors {cfg.num_survivors}, battery {cfg.battery}, max ticks {cfg.max_ticks}")
    print(f"  weights: pheromone={cfg.pheromone_weight}, spread={cfg.spread_weight}, randomness={cfg.randomness}")
    evap = f"on (rate {cfg.evaporation_rate})" if cfg.use_evaporation else "off"
    pings = f"on (range {cfg.ping_range})" if cfg.use_pings else "off"
    print(f"  adaptive features: pheromone evaporation {evap}, survivor acoustic pings {pings}")
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
            for key in ("coverage", "survivors", "first_survivor", "t90", "energy", "collisions", "fitness"):
                print(f"  {key:>10}: {held_before[key]:9.3f} -> {held_after[key]:9.3f}")
            if not generalizes(held_before["fitness"], held_after["fitness"]):
                print("Over-fitting guard: tuned weights score lower on held-out maps, "
                      "so the advisor keeps the default weights.")
                pso = None
                final_cfg, final_results = cfg, base

    print("\nMission Advisor:")
    for rec in advise(final_results, final_cfg, args.priority, pso):
        print(f"  - {rec.format()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
