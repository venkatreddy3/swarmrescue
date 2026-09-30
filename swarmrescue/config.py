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
}


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

    def __post_init__(self) -> None:
        """Validate every field, raising ``ValueError`` on bad values."""
        validate_config(self)

    def with_updates(self, **changes: Any) -> "SwarmConfig":
        """Return a validated copy with ``changes`` applied."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return asdict(self)


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` when ``condition`` is false."""
    if not condition:
        raise ValueError(message)


def validate_config(cfg: SwarmConfig) -> None:
    """Check that ``cfg`` describes a feasible mission.

    Args:
        cfg: Configuration to validate.

    Raises:
        ValueError: If any parameter is out of range or the types are wrong.
    """
    int_fields = (
        "grid_size", "num_agents", "num_survivors", "max_ticks", "battery",
        "comm_range", "sense_range", "shift_tick", "new_walls", "seed",
    )
    for name in int_fields:
        value = getattr(cfg, name)
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"{name} must be an integer, got {value!r}",
        )
    for name in ("wall_density", "pheromone_weight", "spread_weight", "randomness"):
        value = getattr(cfg, name)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{name} must be a number, got {value!r}",
        )

    _require(5 <= cfg.grid_size <= 100, "grid_size must be in [5, 100]")
    _require(0.0 <= cfg.wall_density <= 0.45, "wall_density must be in [0, 0.45]")
    _require(1 <= cfg.num_agents <= 20, "num_agents must be in [1, 20]")
    _require(0 <= cfg.num_survivors <= 50, "num_survivors must be in [0, 50]")
    _require(1 <= cfg.max_ticks <= 5000, "max_ticks must be in [1, 5000]")
    _require(1 <= cfg.battery <= 100000, "battery must be in [1, 100000]")
    _require(1 <= cfg.comm_range <= 2 * cfg.grid_size, "comm_range must be in [1, 2*grid_size]")
    _require(1 <= cfg.sense_range <= 5, "sense_range must be in [1, 5]")
    _require(cfg.seed >= 0, "seed must be non-negative")
    _require(cfg.new_walls >= 0, "new_walls must be non-negative")
    _require(cfg.shift_tick >= 0, "shift_tick must be non-negative")
    for name, (low, high) in TUNABLE_BOUNDS.items():
        value = float(getattr(cfg, name))
        _require(low <= value <= high, f"{name} must be in [{low}, {high}]")

    free_cells_estimate = cfg.grid_size * cfg.grid_size * (1.0 - cfg.wall_density)
    _require(
        cfg.num_agents + cfg.num_survivors <= free_cells_estimate / 4,
        "too many robots + survivors for the free area of this grid",
    )
    _require(
        cfg.new_walls <= cfg.grid_size * cfg.grid_size // 4,
        "new_walls must be at most a quarter of the grid",
    )
