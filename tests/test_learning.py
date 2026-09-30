"""Tests for decentralized online learning (per-robot epsilon-greedy bandits)."""

from __future__ import annotations

import numpy as np
import pytest

from swarmrescue.config import SwarmConfig
from swarmrescue.learning import AdaptiveWeightLearner, weight_presets
from swarmrescue.simulation import MissionControl, SimulationResult


def learner(epsilon: float = 0.0, window: int = 2) -> AdaptiveWeightLearner:
    """A learner over the default presets with a fixed seed."""
    return AdaptiveWeightLearner(weight_presets(1.0, 0.5), epsilon, window, np.random.default_rng(0))


def test_presets_start_from_configured_weights_and_stay_bounded() -> None:
    """Arm 0 is the configured (PSO-tuned) weights; every preset is within PSO bounds."""
    presets = weight_presets(2.5, 2.8)
    assert presets[0] == (2.5, 2.8)
    assert all(0.0 <= w <= 3.0 for pair in presets for w in pair)


def test_reward_is_new_cells_per_move() -> None:
    """A window's reward is its energy efficiency: new cells searched divided by moves."""
    ml = learner(window=4)
    for moved, new in ((True, True), (True, False), (True, True), (False, False)):
        ml.observe(moved, new)
    assert ml.counts[0] == 1 and ml.values[0] == pytest.approx(2 / 3)


def test_greedy_learner_tries_every_preset_then_exploits_the_best() -> None:
    """With epsilon=0 it tries each untried preset once, then keeps the best one."""
    ml = learner(window=1)
    rewards = {0: False, 1: True, 2: False, 3: False}  # only preset 1 finds new cells
    for _ in range(12):
        ml.observe(True, rewards[ml.arm])
    assert all(n >= 1 for n in ml.counts)
    assert ml.arm == 1 and ml.weights == weight_presets(1.0, 0.5)[1]


def test_each_robot_learns_independently() -> None:
    """Decentralized: every robot owns its own learner and generator (no shared, central learner)."""
    mission = MissionControl(SwarmConfig(seed=1, use_learning=True))
    learners = [r.learner for r in mission.robots]
    assert all(ml is not None for ml in learners)
    assert len({id(ml) for ml in learners}) == len(learners)
    assert len({id(ml.rng) for ml in learners if ml is not None}) == len(learners)
    assert all(r.learner is None for r in MissionControl(SwarmConfig(seed=1, use_learning=False)).robots)


def efficiency(results: list[SimulationResult]) -> float:
    """Mean new cells searched per move."""
    return float(np.mean([r.visited_cells / max(1, r.energy_moves) for r in results]))


def test_learning_is_safe_deterministic_and_energy_efficient() -> None:
    """Online learning keeps zero collisions, is reproducible, and does not lower energy efficiency."""
    on = [MissionControl(SwarmConfig(seed=s, use_learning=True), 2).run() for s in (1, 2, 3)]
    off = [MissionControl(SwarmConfig(seed=s, use_learning=False), 2).run() for s in (1, 2, 3)]
    assert all(r.collisions == 0 for r in on)
    again = MissionControl(SwarmConfig(seed=1, use_learning=True), 2).run()
    assert again.fitness == on[0].fitness
    assert efficiency(on) >= efficiency(off)


@pytest.mark.parametrize("bad", [{"learning_epsilon": 1.5}, {"learning_window": 0}, {"use_learning": "yes"}])
def test_learning_parameters_are_validated(bad: dict[str, object]) -> None:
    """Learning settings are bounded like every other field."""
    with pytest.raises(ValueError):
        SwarmConfig(**bad)  # type: ignore[arg-type]
