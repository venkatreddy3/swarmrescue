"""Decentralized online learning: each robot tunes its own behaviour weights while it searches.

There is no central learner and no single point of failure: every
:class:`~swarmrescue.agent.Robot` owns an :class:`AdaptiveWeightLearner`, an
epsilon-greedy multi-armed bandit over a few ``(pheromone_weight,
spread_weight)`` presets. After each short window of ticks the robot rewards
the preset it used with its **energy efficiency** in that window: newly
searched cells per unit of energy (moves). Presets that search more new cells
per move are chosen more often; with probability ``epsilon`` the robot keeps
exploring the other presets, so it can adapt when the situation changes (for
example after an aftershock cuts the short-range radio).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Weights = tuple[float, float]


def weight_presets(pheromone_weight: float, spread_weight: float) -> tuple[Weights, ...]:
    """Arms of the bandit: the configured weights plus three contrasting behaviours.

    Arm 0 is the configured (possibly PSO-tuned) weights, so a learner starts
    from the offline optimum and only moves away when the online reward says so.
    """
    return (
        (pheromone_weight, spread_weight),
        (min(3.0, pheromone_weight * 2.0 + 0.5), spread_weight),  # stronger pheromone trail: avoid searched cells
        (pheromone_weight, min(3.0, spread_weight + 1.0)),  # spread out from teammates
        (pheromone_weight * 0.5, spread_weight * 0.5),  # weaker penalties: shorter path length
    )


@dataclass
class AdaptiveWeightLearner:
    """Epsilon-greedy bandit over weight presets, rewarded by new cells per move.

    Attributes:
        presets: Candidate ``(pheromone_weight, spread_weight)`` pairs.
        epsilon: Exploration probability when a window closes.
        window: Ticks per reward window.
        rng: The robot's own seeded generator (independent of its move noise).
        arm: Index of the preset currently in use.
        values: Running mean reward of each preset.
        counts: Completed windows per preset.
    """

    presets: tuple[Weights, ...]
    epsilon: float
    window: int
    rng: np.random.Generator
    arm: int = 0
    values: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    _ticks: int = 0
    _new_cells: int = 0
    _moves: int = 0

    def __post_init__(self) -> None:
        """Initialise one value estimate and counter per preset."""
        self.values = [0.0] * len(self.presets)
        self.counts = [0] * len(self.presets)

    @property
    def weights(self) -> Weights:
        """``(pheromone_weight, spread_weight)`` the robot should use now."""
        return self.presets[self.arm]

    def observe(self, moved: bool, new_cell: bool) -> None:
        """Record one tick; closes the window every ``window`` ticks."""
        self._ticks += 1
        self._moves += int(moved)
        self._new_cells += int(new_cell)
        if self._ticks >= self.window:
            self._close_window()

    def _close_window(self) -> None:
        """Reward the current preset with its energy efficiency, then pick the next one."""
        reward = self._new_cells / max(1, self._moves)
        self.counts[self.arm] += 1
        self.values[self.arm] += (reward - self.values[self.arm]) / self.counts[self.arm]
        self._ticks = self._new_cells = self._moves = 0
        self.arm = self._select()

    def _select(self) -> int:
        """Epsilon-greedy choice; untried presets are tried first after exploration."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.presets)))
        untried = [i for i, n in enumerate(self.counts) if n == 0]
        if untried:
            return untried[0]
        return int(np.argmax(self.values))
