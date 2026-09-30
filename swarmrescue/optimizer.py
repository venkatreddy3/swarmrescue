"""Particle Swarm Optimisation of the swarm's behaviour weights.

PSO tunes ``(pheromone_weight, spread_weight, randomness)`` to maximise the
mission fitness averaged over several seeds (different buildings), so the
tuned weights generalise instead of over-fitting one map.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from swarmrescue.config import BEHAVIOUR_WEIGHTS, TUNABLE_BOUNDS, SwarmConfig
from swarmrescue.simulation import mean_fitness

PARAM_NAMES: tuple[str, ...] = BEHAVIOUR_WEIGHTS


def search_space(cfg: SwarmConfig) -> tuple[str, ...]:
    """Parameters PSO tunes for ``cfg``.

    These are the three behaviour weights, plus ``evaporation_rate`` when
    pheromone evaporation is enabled.
    """
    return PARAM_NAMES + (("evaporation_rate",) if cfg.use_evaporation else ())


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


def _to_params(position: np.ndarray, names: Sequence[str] = PARAM_NAMES) -> dict[str, float]:
    """Convert a particle position vector to a parameter dictionary."""
    return {name: float(round(v, 6)) for name, v in zip(names, position, strict=True)}


def objective(
    position: np.ndarray,
    base_cfg: SwarmConfig,
    seeds: Sequence[int],
    round_no: int,
    names: Sequence[str] = PARAM_NAMES,
) -> float:
    """Mean mission fitness of the parameters encoded by ``position``."""
    return mean_fitness(base_cfg.with_updates(**_to_params(position, names)), tuple(seeds), round_no)


class _Swarm:
    """Particle positions, velocities and personal/global bests for one PSO run."""

    def __init__(self, base_cfg: SwarmConfig, n_particles: int, rng: np.random.Generator) -> None:
        """Scatter particles uniformly in the bounds; particle 0 sits at ``base_cfg``'s weights."""
        self.names = search_space(base_cfg)
        self.low = np.array([TUNABLE_BOUNDS[p][0] for p in self.names])
        self.high = np.array([TUNABLE_BOUNDS[p][1] for p in self.names])
        span = self.high - self.low
        self.vmax = 0.2 * span
        self.pos = self.low + rng.random((n_particles, len(self.names))) * span
        self.pos[0] = [float(getattr(base_cfg, p)) for p in self.names]
        self.vel = (rng.random(self.pos.shape) * 2 - 1) * 0.1 * span
        self.fit = np.zeros(n_particles)
        self.pbest, self.pbest_fit = self.pos.copy(), np.full(n_particles, -np.inf)
        self.gbest, self.gbest_fit = self.pos[0].copy(), -np.inf

    def evaluate(self, fitness: Callable[[np.ndarray], float]) -> None:
        """Score every particle and update personal and global bests."""
        self.fit = np.array([fitness(p) for p in self.pos])
        improved = self.fit > self.pbest_fit
        self.pbest[improved], self.pbest_fit[improved] = self.pos[improved], self.fit[improved]
        g = int(np.argmax(self.pbest_fit))
        if self.pbest_fit[g] > self.gbest_fit:
            self.gbest, self.gbest_fit = self.pbest[g].copy(), float(self.pbest_fit[g])

    def move(self, rng: np.random.Generator, inertia: float, c1: float, c2: float) -> None:
        """Standard velocity update, clamped to 20% of the range; positions clipped to bounds."""
        r1, r2 = rng.random(self.pos.shape), rng.random(self.pos.shape)
        vel = inertia * self.vel + c1 * r1 * (self.pbest - self.pos) + c2 * r2 * (self.gbest - self.pos)
        self.vel = np.clip(vel, -self.vmax, self.vmax)
        self.pos = np.clip(self.pos + self.vel, self.low, self.high)

    def log(self, iteration: int) -> IterationLog:
        """Convergence record for the current state."""
        return IterationLog(iteration, self.gbest_fit, float(self.fit.mean()), _to_params(self.gbest, self.names))


@dataclass(frozen=True)
class PSOOptimizer:
    """Global-best Particle Swarm Optimisation of the swarm's behaviour weights.

    The optimiser itself is offline tuning; the over-fitting guard in ``main.py``
    re-checks its result on unseen maps before the Mission Advisor recommends it.
    """

    n_particles: int = 8
    n_iters: int = 12
    inertia: float = 0.6
    c1: float = 1.5
    c2: float = 1.5
    rng_seed: int = 0

    def optimize(
        self,
        base_cfg: SwarmConfig,
        round_no: int = 1,
        seeds: Sequence[int] = (1, 2, 3),
        callback: Callable[[IterationLog], None] | None = None,
    ) -> PSOResult:
        """Maximise mean fitness over ``seeds``; the best-fitness history never decreases."""
        if self.n_particles < 1 or self.n_iters < 0 or not seeds:
            raise ValueError("need n_particles >= 1, n_iters >= 0 and at least one seed")
        rng = np.random.default_rng(self.rng_seed)
        swarm = _Swarm(base_cfg, self.n_particles, rng)

        def fitness(position: np.ndarray) -> float:
            """Objective for one particle."""
            return objective(position, base_cfg, seeds, round_no, swarm.names)

        swarm.evaluate(fitness)
        baseline = float(swarm.fit[0])
        history = [swarm.log(0)]
        for it in range(1, self.n_iters + 1):
            if callback:
                callback(history[-1])
            swarm.move(rng, self.inertia, self.c1, self.c2)
            swarm.evaluate(fitness)
            history.append(swarm.log(it))
        if callback:
            callback(history[-1])
        return PSOResult(_to_params(swarm.gbest, swarm.names), swarm.gbest_fit, baseline, tuple(history))


def run_pso(
    base_cfg: SwarmConfig,
    round_no: int = 1,
    seeds: Sequence[int] = (1, 2, 3),
    *,
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
    optimizer = PSOOptimizer(n_particles, n_iters, inertia, c1, c2, rng_seed)
    return optimizer.optimize(base_cfg, round_no, seeds, callback)
