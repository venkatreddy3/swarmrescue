"""Particle Swarm Optimisation of the swarm's behaviour weights.

PSO tunes ``(pheromone_weight, spread_weight, randomness)`` to maximise the
mission fitness averaged over several seeds (different buildings), so the
tuned weights generalise instead of over-fitting one map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from swarmrescue.config import TUNABLE_BOUNDS, SwarmConfig
from swarmrescue.simulation import mean_fitness

PARAM_NAMES: tuple[str, ...] = tuple(TUNABLE_BOUNDS)


@dataclass(frozen=True)
class IterationLog:
    """Convergence record for one PSO iteration.

    Attributes:
        iteration: 0 for the initial swarm, then 1..n_iters.
        best_fitness: Global best fitness found so far.
        mean_fitness: Mean fitness of the particles evaluated this iteration.
        best_params: Parameters of the global best so far.
    """

    iteration: int
    best_fitness: float
    mean_fitness: float
    best_params: dict[str, float]

    def format(self) -> str:
        """One human-readable convergence-log line."""
        params = ", ".join(f"{k}={v:.3f}" for k, v in self.best_params.items())
        return (
            f"[PSO] iter {self.iteration:2d} | best {self.best_fitness:8.3f} | "
            f"swarm mean {self.mean_fitness:8.3f} | {params}"
        )


@dataclass(frozen=True)
class PSOResult:
    """Outcome of a PSO run."""

    best_params: dict[str, float]
    best_fitness: float
    baseline_fitness: float
    history: tuple[IterationLog, ...]

    def best_config(self, base: SwarmConfig) -> SwarmConfig:
        """Return ``base`` with the tuned parameters applied."""
        return base.with_updates(**self.best_params)


def _to_params(position: np.ndarray) -> dict[str, float]:
    """Convert a particle position vector to a parameter dictionary."""
    return {name: float(round(v, 6)) for name, v in zip(PARAM_NAMES, position)}


def objective(
    position: np.ndarray, base_cfg: SwarmConfig, seeds: Sequence[int], round_no: int
) -> float:
    """Mean mission fitness of the parameters encoded by ``position``."""
    return mean_fitness(base_cfg.with_updates(**_to_params(position)), tuple(seeds), round_no)


def run_pso(
    base_cfg: SwarmConfig,
    round_no: int = 1,
    seeds: Sequence[int] = (1, 2, 3),
    n_particles: int = 8,
    n_iters: int = 12,
    inertia: float = 0.6,
    c1: float = 1.5,
    c2: float = 1.5,
    rng_seed: int = 0,
    callback: Callable[[IterationLog], None] | None = None,
) -> PSOResult:
    """Maximise mean fitness with global-best Particle Swarm Optimisation.

    Particle 0 starts at the current (hand-set) weights, so the result is
    never worse than the baseline on the tuning seeds. Velocities are clamped
    to 20% of each parameter range and positions are clipped to the bounds.

    Args:
        base_cfg: Mission configuration whose weights are tuned.
        round_no: Scenario to optimise for (1 or 2).
        seeds: Seeds (maps) the fitness is averaged over.
        n_particles: Swarm size.
        n_iters: Number of velocity/position updates.
        inertia: Inertia weight ``w``.
        c1: Cognitive (personal-best) coefficient.
        c2: Social (global-best) coefficient.
        rng_seed: Seed for the optimiser's own randomness.
        callback: Called with each :class:`IterationLog` (convergence log).

    Returns:
        A :class:`PSOResult` with a monotone non-decreasing best history.

    Raises:
        ValueError: On invalid swarm settings.
    """
    if n_particles < 1 or n_iters < 0 or not seeds:
        raise ValueError("need n_particles >= 1, n_iters >= 0 and at least one seed")
    rng = np.random.default_rng(rng_seed)
    low = np.array([TUNABLE_BOUNDS[p][0] for p in PARAM_NAMES])
    high = np.array([TUNABLE_BOUNDS[p][1] for p in PARAM_NAMES])
    span = high - low
    vmax = 0.2 * span

    pos = low + rng.random((n_particles, len(PARAM_NAMES))) * span
    pos[0] = [float(getattr(base_cfg, p)) for p in PARAM_NAMES]
    vel = (rng.random(pos.shape) * 2 - 1) * 0.1 * span
    fit = np.array([objective(p, base_cfg, seeds, round_no) for p in pos])
    baseline = float(fit[0])

    pbest, pbest_fit = pos.copy(), fit.copy()
    g = int(np.argmax(pbest_fit))
    gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
    history = [IterationLog(0, gbest_fit, float(fit.mean()), _to_params(gbest))]
    if callback:
        callback(history[-1])

    for it in range(1, n_iters + 1):
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = inertia * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, low, high)
        fit = np.array([objective(p, base_cfg, seeds, round_no) for p in pos])
        improved = fit > pbest_fit
        pbest[improved], pbest_fit[improved] = pos[improved], fit[improved]
        g = int(np.argmax(pbest_fit))
        if pbest_fit[g] > gbest_fit:
            gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
        history.append(IterationLog(it, gbest_fit, float(fit.mean()), _to_params(gbest)))
        if callback:
            callback(history[-1])

    return PSOResult(
        best_params=_to_params(gbest),
        best_fitness=gbest_fit,
        baseline_fitness=baseline,
        history=tuple(history),
    )
