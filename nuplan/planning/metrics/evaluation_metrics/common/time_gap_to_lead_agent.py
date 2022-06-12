from typing import Dict, List, Optional

import numpy as np

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.utils import get_closest_agent_in_position, is_agent_ahead


class TimeGapToLeadAgent(MetricBase):
    """Time Gap to lead agents metric."""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializes the TimeGapToLeadAgent class
        :param name: Metric name
        :param category: Metric category.
        """
        super().__init__(name=name, category=category)

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        # TODO: Define the metric score
        return 0.0

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        time_gap = []
        for sample in history.data:
            ego_state = sample.ego_state
            observations = sample.observation
            closest_agent, closest_distance = get_closest_agent_in_position(ego_state, observations, is_agent_ahead)

            # clamp to avoid zero division
            time_gap_value = closest_distance / (max(ego_state.dynamic_car_state.speed, 0.1))
            time_gap.append(time_gap_value)

        # Extract timestamps and remove nan values
        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_gap = np.asarray(time_gap)  # type: ignore
        time_gap = time_gap[np.isfinite(time_gap)]

        # If there are no values, we skip.
        if not len(time_gap):
            return []

        # Insert nan values
        nan_values = [np.nan] * (len(time_stamps) - len(time_gap))
        time_gap = np.insert(time_gap, len(time_gap), nan_values)  # type: ignore
        time_series = TimeSeries(unit='seconds', time_stamps=list(time_stamps), values=time_gap)

        metric_statistics = self._compute_time_series_statistic(time_series=time_series)
        results = self._construct_metric_results(
            metric_statistics=metric_statistics, time_series=time_series, scenario=scenario
        )
        return results  # type: ignore
