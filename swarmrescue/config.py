"""Mission configuration for SwarmRescue.

All tunable parameters live in a single frozen dataclass so that a run is fully
described by (config, seed) and can never be mutated mid-simulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

# Search bounds used by the PSO optimiser: name -> (low, high).
TUNABLE_BOUNDS: dict[str, tuple[float, float]] = {
    "pheromone_weight": (0.0, 3.0),
    "spread_weight": (0.0, 3.0),
    "randomness": (0.0, 1.0),
    "evaporation_rate": (0.0, 0.2),
}
# Parameters PSO always tunes; evaporation_rate is added when evaporation is on.
BEHAVIOUR_WEIGHTS: tuple[str, ...] = ("pheromone_weight", "spread_weight", "randomness")


@dataclass(frozen=True)
class SwarmConfig:
    """Immutable parameter set for one SwarmRescue mission.

    Attributes:
        grid_size: Side length of the square occupancy grid.
        wall_density: Fraction of cells initially filled with debris/walls.
        num_agents: Number of rescue robots in the swarm.
        num_survivors: Number of trapped survivors hidden in reachable cells.
        max_ticks: Mission time limit in simulation ticks.
        battery: Moves each robot can make before it stops.
        comm_range: Manhattan radius for map sharing / seeing teammates.
        sense_range: Chebyshev radius of each robot's wall sensor.
        pheromone_weight: Penalty per recorded visit of a candidate cell.
        spread_weight: Penalty for moving close to other robots.
        randomness: Amplitude of uniform noise added to move scores.
        shift_tick: Tick at which the Round 2 scenario shift happens.
        new_walls: Number of debris cells that appear at the shift.
        seed: Master random seed (map, survivors, noise, shift).
        use_evaporation: Enable pheromone evaporation (trails fade over time).
        evaporation_rate: Fraction of pheromone lost per tick when enabled.
        use_pings: Enable survivor acoustic pings (tapping / phone signals).
        ping_range: Manhattan distance at which a robot hears a survivor.
        latency_budget_ms: Real-time budget for one tick of swarm decisions.
        safety_margin: Minimum Manhattan clearance a robot keeps from other
            robots when it can (0 = only never share a cell).
        use_learning: Decentralized online learning: each robot runs its own
            epsilon-greedy bandit over behaviour-weight presets.
        learning_epsilon: Exploration probability of each robot's bandit.
        learning_window: Ticks per reward window (new cells per move).
    """

    grid_size: int = 20
    wall_density: float = 0.18
    num_agents: int = 4
    num_survivors: int = 5
    max_ticks: int = 300
    battery: int = 250
    comm_range: int = 5
    sense_range: int = 1
    pheromone_weight: float = 1.0
    spread_weight: float = 0.5
    randomness: float = 0.1
    shift_tick: int = 60
    new_walls: int = 25
    seed: int = 0
    use_evaporation: bool = False
    evaporation_rate: float = 0.01
    use_pings: bool = True
    ping_range: int = 4
    latency_budget_ms: float = 50.0
    safety_margin: int = 0
    use_learning: bool = True
    learning_epsilon: float = 0.1
    learning_window: int = 10

    def __post_init__(self) -> None:
        """Validate every field, raising ``ValueError`` on bad values."""
        validate_config(self)

    def with_updates(self, **changes: Any) -> SwarmConfig:
        """Return a validated copy with ``changes`` applied."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return asdict(self)


# Inclusive bounds for every numeric field. comm_range is further limited to 2*grid_size.
INT_BOUNDS: dict[str, tuple[int, int]] = {
    "grid_size": (5, 100),
    "num_agents": (1, 20),
    "num_survivors": (0, 50),
    "max_ticks": (1, 5000),
    "battery": (1, 100_000),
    "comm_range": (1, 200),
    "sense_range": (1, 5),
    "shift_tick": (0, 5000),
    "new_walls": (0, 2500),
    "seed": (0, 2**32 - 1),
    "ping_range": (1, 10),
    "safety_margin": (0, 3),
    "learning_window": (1, 200),
}
FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "wall_density": (0.0, 0.45),
    "latency_budget_ms": (1.0, 1000.0),
    "learning_epsilon": (0.0, 1.0),
    **TUNABLE_BOUNDS,
}
BOOL_FIELDS: tuple[str, ...] = ("use_evaporation", "use_pings", "use_learning")


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` when ``condition`` is false."""
    if not condition:
        raise ValueError(message)


def _validate_types(cfg: SwarmConfig) -> None:
    """Every field has the right type (booleans are not accepted as numbers)."""
    for name in INT_BOUNDS:
        value = getattr(cfg, name)
        _require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer, got {value!r}")
    for name in FLOAT_BOUNDS:
        value = getattr(cfg, name)
        _require(
            isinstance(value, int | float) and not isinstance(value, bool), f"{name} must be a number, got {value!r}"
        )
    for name in BOOL_FIELDS:
        _require(isinstance(getattr(cfg, name), bool), f"{name} must be True or False")


def _validate_ranges(cfg: SwarmConfig) -> None:
    """Every numeric field lies inside its inclusive bounds (NaN is rejected)."""
    for name, (lo, hi) in {**INT_BOUNDS, **FLOAT_BOUNDS}.items():
        value = getattr(cfg, name)
        _require(lo <= value <= hi, f"{name} must be in [{lo}, {hi}]")


def _validate_feasibility(cfg: SwarmConfig) -> None:
    """Cross-field checks: the mission must physically fit in the grid."""
    area = cfg.grid_size * cfg.grid_size
    _require(cfg.comm_range <= 2 * cfg.grid_size, "comm_range must be in [1, 2*grid_size]")
    _require(
        cfg.num_agents + cfg.num_survivors <= area * (1.0 - cfg.wall_density) / 4,
        "too many robots + survivors for the free area of this grid",
    )
    _require(cfg.new_walls <= area // 4, "new_walls must be at most a quarter of the grid")


def validate_config(cfg: SwarmConfig) -> None:
    """Check that ``cfg`` describes a feasible mission.

    Args:
        cfg: Configuration to validate.

    Raises:
        ValueError: If any parameter has the wrong type, is out of range, or the
            combination cannot fit in the grid.
    """
    _validate_types(cfg)
    _validate_ranges(cfg)
    _validate_feasibility(cfg)
