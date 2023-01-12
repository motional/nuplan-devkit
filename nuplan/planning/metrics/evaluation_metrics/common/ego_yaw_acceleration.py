from typing import List

from nuplan.planning.metrics.evaluation_metrics.base.within_bound_metric_base import WithinBoundMetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics
from nuplan.planning.metrics.utils.state_extractors import extract_ego_yaw_rate
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoYawAccelerationStatistics(WithinBoundMetricBase):
    """Ego yaw acceleration metric."""

    def __init__(self, name: str, category: str, max_abs_yaw_accel: float) -> None:
        """
        Initializes the EgoYawAccelerationStatistics class
        :param name: Metric name
        :param category: Metric category
        :param max_abs_yaw_accel: Maximum threshold to define if absolute yaw acceleration is within bound.
        """
        super().__init__(name=name, category=category)
        self._max_abs_yaw_accel = max_abs_yaw_accel

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the yaw acceleration metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated yaw acceleration metric.
        """
        metric_statistics: List[MetricStatistics] = self._compute_statistics(
            history=history,
            scenario=scenario,
            statistic_unit_name='radians_per_second_squared',
            extract_function=extract_ego_yaw_rate,
            extract_function_params={'deriv_order': 2, 'poly_order': 3},
            min_within_bound_threshold=-self._max_abs_yaw_accel,
            max_within_bound_threshold=self._max_abs_yaw_accel,
        )
        return metric_statistics
