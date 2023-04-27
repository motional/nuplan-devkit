from typing import Any, List

import numpy as np

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic
from nuplan.planning.metrics.utils.state_extractors import extract_ego_velocity
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoMeanSpeedStatistics(MetricBase):
    """Ego mean speed metric."""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializes the EgoMeanSpeedStatistics class
        :param name: Metric name
        :param category: Metric category.
        """
        super().__init__(name=name, category=category)

    @staticmethod
    def ego_avg_speed(history: SimulationHistory) -> Any:
        """
        Compute mean of ego speed over the scenario duration
        :param history: History from a simulation engine
        :return mean of ego speed (m/s).
        """
        ego_states = history.extract_ego_state
        ego_velocities = extract_ego_velocity(ego_states)
        mean_speed = np.mean(ego_velocities)

        return mean_speed

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the mean of ego speed over the scenario duration
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the mean of ego speed.
        """
        mean_speed = self.ego_avg_speed(history=history)

        statistics = [
            Statistic(
                name='ego_mean_speed_value', unit='meters_per_second', value=mean_speed, type=MetricStatisticsType.VALUE
            )
        ]

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=statistics, time_series=None, scenario=scenario
        )
        return results
