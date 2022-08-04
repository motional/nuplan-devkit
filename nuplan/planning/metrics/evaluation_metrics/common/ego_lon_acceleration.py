from typing import List

from nuplan.planning.metrics.evaluation_metrics.base.within_bound_metric_base import WithinBoundMetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics
from nuplan.planning.metrics.utils.state_extractors import extract_ego_acceleration
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoLonAccelerationStatistics(WithinBoundMetricBase):
    """Ego longitudinal acceleration metric."""

    def __init__(self, name: str, category: str, min_lon_accel: float, max_lon_accel: float) -> None:
        """
        Initializes the EgoLonAccelerationStatistics class
        :param name: Metric name
        :param category: Metric category
        :param min_lon_accel: Threshold to define if the lon acceleration is within bound
        :param max_lon_accel: Threshold to define if the lat acceleration is within bound.
        """
        super().__init__(name=name, category=category)
        self._min_lon_accel = min_lon_accel
        self._max_lon_accel = max_lon_accel

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the longitudinal acceleration metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated longitudinal acceleration metric.
        """
        metric_statistics: List[MetricStatistics] = self._compute_statistics(
            history=history,
            scenario=scenario,
            statistic_unit_name='meters_per_second_squared',
            extract_function=extract_ego_acceleration,
            extract_function_params={'acceleration_coordinate': 'x'},
            min_within_bound_threshold=self._min_lon_accel,
            max_within_bound_threshold=self._max_lon_accel,
        )
        return metric_statistics
