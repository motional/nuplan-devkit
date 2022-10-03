from dataclasses import dataclass
from typing import List, Optional, Sequence, Union


@dataclass(frozen=True)
class ScenarioFilter:
    """
    Collection of filters used to construct scenarios from a database for training/simulation.
    """

    scenario_types: Optional[List[str]]  # List of scenario types to include
    scenario_tokens: Optional[List[Sequence[str]]]  # List of scenarios to include in the form of (log_name, token)

    log_names: Optional[List[str]]  # Filter scenarios by log names
    map_names: Optional[List[str]]  # Filter scenarios by map names

    num_scenarios_per_type: Optional[int]  # Number of scenarios per type
    limit_total_scenarios: Optional[Union[int, float]]  # Limit total scenarios (float = fraction, int = num)
    timestamp_threshold_s: Optional[
        float
    ]  # Threshold for the interval of time between scenario initial lidar timestamps in seconds

    expand_scenarios: bool  # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals: bool  # Whether to remove scenarios where the mission goal is invalid
    shuffle: bool  # Whether to shuffle the scenarios

    def __post_init__(self) -> None:
        """Sanitize class attributes."""
        if self.num_scenarios_per_type is not None:
            assert 0 < self.num_scenarios_per_type, "num_scenarios_per_type should be a positive integer"

        if isinstance(self.limit_total_scenarios, float):
            assert 0.0 < self.limit_total_scenarios <= 1.0, "limit_total_scenarios should be in (0, 1] when float"
        elif isinstance(self.limit_total_scenarios, int):
            assert 0 < self.limit_total_scenarios, "limit_total_scenarios should be positive when integer"
