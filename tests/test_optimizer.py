"""Tests for the Particle Swarm Optimiser."""

from __future__ import annotations

import itertools
from itertools import pairwise

import pytest

from swarmrescue.config import TUNABLE_BOUNDS, SwarmConfig
from swarmrescue.optimizer import IterationLog, PSOResult, run_pso
from swarmrescue.simulation import mean_fitness

SEEDS = (1, 2)


@pytest.fixture(scope="module")
def tiny_cfg() -> SwarmConfig:
    """A small mission so PSO tests run in seconds."""
    return SwarmConfig(
        grid_size=10, num_agents=2, num_survivors=2, max_ticks=80, battery=80, shift_tick=20, new_walls=5
    )


@pytest.fixture(scope="module")
def pso_run(tiny_cfg: SwarmConfig) -> tuple[PSOResult, list[IterationLog]]:
    """One short PSO run plus the convergence log it emitted."""
    logs: list[IterationLog] = []
    result = run_pso(tiny_cfg, seeds=SEEDS, n_particles=5, n_iters=4, rng_seed=3, callback=logs.append)
    return result, logs


def test_best_fitness_never_decreases(pso_run: tuple[PSOResult, list[IterationLog]]) -> None:
    """Global best is monotone non-decreasing across iterations."""
    result, _ = pso_run
    bests = [h.best_fitness for h in result.history]
    assert all(b2 >= b1 for b1, b2 in itertools.pairwise(bests))
    assert result.best_fitness == bests[-1] == max(bests)


def test_never_worse_than_baseline(pso_run: tuple[PSOResult, list[IterationLog]]) -> None:
    """Particle 0 starts at the default weights, so tuning cannot lose."""
    result, _ = pso_run
    assert result.best_fitness >= result.baseline_fitness


def test_params_within_bounds_and_reproducible(
    pso_run: tuple[PSOResult, list[IterationLog]], tiny_cfg: SwarmConfig
) -> None:
    """Best params respect bounds and re-evaluate to the reported fitness."""
    result, _ = pso_run
    for name, value in result.best_params.items():
        low, high = TUNABLE_BOUNDS[name]
        assert low <= value <= high
    tuned = result.best_config(tiny_cfg)
    assert mean_fitness(tuned, SEEDS) == pytest.approx(result.best_fitness, abs=1e-6)


def test_convergence_log_emitted(pso_run: tuple[PSOResult, list[IterationLog]]) -> None:
    """Callback receives one log per iteration (plus the initial swarm)."""
    result, logs = pso_run
    assert [log.iteration for log in logs] == list(range(5))
    assert list(result.history) == logs
    assert "best" in logs[-1].format() and "pheromone_weight" in logs[-1].format()


def test_pso_is_deterministic(tiny_cfg: SwarmConfig) -> None:
    """Same optimiser seed gives the same answer."""
    a = run_pso(tiny_cfg, seeds=(1,), n_particles=3, n_iters=2, rng_seed=5)
    b = run_pso(tiny_cfg, seeds=(1,), n_particles=3, n_iters=2, rng_seed=5)
    assert a.best_params == b.best_params and a.best_fitness == b.best_fitness


def test_pso_rejects_bad_settings(tiny_cfg: SwarmConfig) -> None:
    """Empty swarm or no seeds is invalid."""
    with pytest.raises(ValueError):
        run_pso(tiny_cfg, n_particles=0)
    with pytest.raises(ValueError):
        run_pso(tiny_cfg, seeds=())


def test_pso_tunes_evaporation_rate_when_enabled(tiny_cfg: SwarmConfig) -> None:
    """With evaporation on, PSO returns a bounded evaporation_rate too."""
    result = run_pso(tiny_cfg.with_updates(use_evaporation=True), seeds=(1,), n_particles=3, n_iters=1, rng_seed=1)
    low, high = TUNABLE_BOUNDS["evaporation_rate"]
    assert low <= result.best_params["evaporation_rate"] <= high
    bests = [h.best_fitness for h in result.history]
    assert all(b >= a for a, b in pairwise(bests))
