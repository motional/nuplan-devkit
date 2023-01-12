from typing import List

from nuplan.planning.metrics.evaluation_metrics.base.within_bound_metric_base import WithinBoundMetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics
from nuplan.planning.metrics.utils.state_extractors import extract_ego_yaw_rate
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoYawRateStatistics(WithinBoundMetricBase):
    """Ego yaw rate metric."""

    def __init__(self, name: str, category: str, max_abs_yaw_rate: float) -> None:
        """
        Initializes the EgoYawRateStatistics class
        :param name: Metric name
        :param category: Metric category
        :param max_abs_yaw_rate: Maximum threshold to define if absolute yaw rate is within bound.
        """
        super().__init__(name=name, category=category)
        self._max_abs_yaw_rate = max_abs_yaw_rate

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the yaw rate  metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated yaw rate metric.
        """
        metric_statistics: List[MetricStatistics] = self._compute_statistics(
            history=history,
            scenario=scenario,
            statistic_unit_name='radians_per_second',
            extract_function=extract_ego_yaw_rate,
            extract_function_params={},
            min_within_bound_threshold=-self._max_abs_yaw_rate,
            max_within_bound_threshold=self._max_abs_yaw_rate,
        )
        return metric_statistics
