from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass(frozen=True)
class ScenarioFilters:
    # Include all scenarios from following log names
    log_names: Optional[List[str]]
    # Include all scenarios from logs with the following tags
    log_labels: Optional[List[str]]
    # If log_names is present, number of scenarios from a log
    max_scenarios_per_log: Optional[int]
    # List of scenario types to include
    scenario_types: Optional[List[str]]
    # List of start/end sample tokens which will be always included
    scenario_tokens: Optional[List[Tuple[str, str]]]
    # Only for this map scenarios will be extracted
    map_name: Optional[str]

    # If True, shuffle the scenarios
    shuffle: bool
    # Limit scenarios per scenario type (float = fraction, int = num)
    limit_scenarios_per_type: Optional[Union[int, float]]
    # Subsample a scenario relative to the database frequency
    subsample_ratio: Optional[float]
    # Convert the final multi-sample scenario list to a list of single-sample scenarios
    flatten_scenarios: bool
    # If True, remove scenarios where the mission goal is invalid
    remove_invalid_goals: bool
    # Limit total scenarios (float = fraction, int = num)
    limit_total_scenarios: Optional[Union[int, float]]

    def __post_init__(self) -> None:
        if self.max_scenarios_per_log:
            assert self.log_names, "max_scenarios_per_log is set, but no log_names were provided!"

        if isinstance(self.limit_scenarios_per_type, float):
            assert 0.0 < self.limit_scenarios_per_type <= 1.0, "limit_scenarios_per_type should be in (0, 1] when float"
        elif isinstance(self.limit_scenarios_per_type, int):
            assert 0 < self.limit_scenarios_per_type, "limit_scenarios_per_type should be positive when integer"

        if isinstance(self.limit_total_scenarios, float):
            assert 0.0 < self.limit_total_scenarios <= 1.0, "limit_total_scenarios should be in (0, 1] when float"
        elif isinstance(self.limit_total_scenarios, int):
            assert 0 < self.limit_total_scenarios, "limit_total_scenarios should be positive when integer"

        if self.subsample_ratio:
            assert 0 < self.subsample_ratio <= 1.0, 'subsample_ratio should be in (0, 1]'
