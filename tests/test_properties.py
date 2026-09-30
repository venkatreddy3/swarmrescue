"""Property-based tests (Hypothesis): invariants that must hold for *any* disaster zone."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from swarmrescue.agent import bfs_path
from swarmrescue.config import SwarmConfig
from swarmrescue.settings import SEED_BOUNDS, load_settings
from swarmrescue.simulation import simulate
from swarmrescue.world import FREE

PROPERTY_SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def mission_configs(draw: st.DrawFn) -> SwarmConfig:
    """Random small but valid missions: size, debris, swarm, survivors, weights, features."""
    grid = draw(st.integers(6, 14))
    kwargs = {
        "grid_size": grid,
        "wall_density": draw(st.floats(0.0, 0.35)),
        "num_agents": draw(st.integers(1, 5)),
        "num_survivors": draw(st.integers(0, 3)),
        "max_ticks": draw(st.integers(20, 120)),
        "battery": draw(st.integers(5, 150)),
        "comm_range": draw(st.integers(1, grid)),
        "pheromone_weight": draw(st.floats(0.0, 3.0)),
        "spread_weight": draw(st.floats(0.0, 3.0)),
        "randomness": draw(st.floats(0.0, 1.0)),
        "shift_tick": draw(st.integers(1, 40)),
        "new_walls": draw(st.integers(0, grid * grid // 4)),
        "seed": draw(st.integers(0, 10_000)),
        "use_pings": draw(st.booleans()),
        "use_evaporation": draw(st.booleans()),
    }
    try:
        return SwarmConfig(**kwargs)  # type: ignore[arg-type]
    except ValueError:
        assume(False)  # infeasible combination (e.g. too many robots for a tiny dense map)
        raise


def true_distance(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Independent shortest distance: vectorised Bellman-Ford relaxation (not BFS)."""
    inf = passable.size + 1
    dist = np.full(passable.shape, inf)
    dist[start] = 0
    while True:
        best = dist.copy()
        best[1:, :] = np.minimum(best[1:, :], dist[:-1, :] + 1)
        best[:-1, :] = np.minimum(best[:-1, :], dist[1:, :] + 1)
        best[:, 1:] = np.minimum(best[:, 1:], dist[:, :-1] + 1)
        best[:, :-1] = np.minimum(best[:, :-1], dist[:, 1:] + 1)
        best[~passable] = inf
        best[start] = 0
        if np.array_equal(best, dist):
            return int(dist[goal]) if dist[goal] < inf else -1
        dist = best


@PROPERTY_SETTINGS
@given(cfg=mission_configs(), round_no=st.sampled_from([1, 2]))
def test_any_mission_is_collision_free_and_bounded(cfg: SwarmConfig, round_no: int) -> None:
    """Zero collisions, robots never on debris or sharing a cell, coverage always in [0, 1]."""
    try:
        result = simulate(cfg, round_no, record_frames=True)
    except RuntimeError:
        assume(False)  # no valid disaster zone exists for these parameters
        raise
    assert result.collisions == 0 and result.interlock_trips == 0
    assert 0.0 <= result.coverage <= 1.0
    assert all(0.0 <= c <= 1.0 for c in result.coverage_curve)
    assert 0 <= result.survivors_found <= result.survivors_total
    assert result.energy_moves <= cfg.battery * cfg.num_agents
    for prev, frame in pairwise(result.frames):
        assert len(set(frame.positions)) == len(frame.positions)
        for old, new in zip(prev.positions, frame.positions, strict=True):
            assert frame.grid[new] == FREE
            assert abs(old[0] - new[0]) + abs(old[1] - new[1]) <= 1


@PROPERTY_SETTINGS
@given(
    passable=arrays(bool, st.tuples(st.integers(2, 12), st.integers(2, 12)), elements=st.booleans()),
    data=st.data(),
)
def test_bfs_path_length_equals_true_shortest_distance(passable: np.ndarray, data: st.DataObject) -> None:
    """BFS returns a valid path whose length is exactly the true shortest distance."""
    side = min(passable.shape)
    grid = passable[:side, :side].copy()
    free = [tuple(int(v) for v in rc) for rc in np.argwhere(grid)]
    if len(free) < 2:
        return
    start, goal = data.draw(st.sampled_from(free)), data.draw(st.sampled_from(free))
    if start == goal:
        return
    goal_mask = np.zeros_like(grid)
    goal_mask[goal] = True
    path = bfs_path(grid, (start[0], start[1]), goal_mask)
    expected = true_distance(grid, (start[0], start[1]), (goal[0], goal[1]))
    if expected < 0:
        assert path is None
        return
    assert path is not None and len(path) - 1 == expected
    assert path[0] == start and path[-1] == goal
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and grid[b] for a, b in pairwise(path))


@settings(max_examples=200, deadline=None)
@given(seed=st.text(max_size=40), level=st.text(max_size=40))
def test_settings_loader_never_crashes_and_stays_bounded(seed: str, level: str) -> None:
    """Arbitrary (even hostile) environment values always yield bounded, valid settings."""
    s = load_settings(env={"SWARM_SEED": seed, "LOG_LEVEL": level}, env_file=None)
    assert SEED_BOUNDS[0] <= s.seed <= SEED_BOUNDS[1]
    assert s.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@settings(max_examples=100, deadline=None)
@given(value=st.one_of(st.floats(allow_nan=True), st.integers(-10, 10), st.text(max_size=5), st.none()))
def test_config_rejects_every_invalid_weight(value: object) -> None:
    """A behaviour weight outside [0, 3] (or not a real number) is always rejected."""
    valid = isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= value <= 3.0
    try:
        SwarmConfig(pheromone_weight=value)  # type: ignore[arg-type]
    except ValueError:
        assert not valid
    else:
        assert valid
