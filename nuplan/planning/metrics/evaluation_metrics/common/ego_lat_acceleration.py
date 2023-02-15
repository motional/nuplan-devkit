from typing import List

import numpy as np

from nuplan.planning.metrics.evaluation_metrics.base.within_bound_metric_base import WithinBoundMetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics
from nuplan.planning.metrics.utils.state_extractors import extract_ego_acceleration
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoLatAccelerationStatistics(WithinBoundMetricBase):
    """Ego lateral acceleration metric."""

    def __init__(self, name: str, category: str, max_abs_lat_accel: float) -> None:
        """
        Initializes the EgoLatAccelerationStatistics class
        :param name: Metric name
        :param category: Metric category
        :param max_abs_lat_accel: Maximum threshold to define if absolute lateral acceleration is within bound.
        """
        super().__init__(name=name, category=category)
        self._max_abs_lat_accel = max_abs_lat_accel

    @staticmethod
    def compute_comfortability(history: SimulationHistory, max_abs_lat_accel: float) -> bool:
        """
        Compute comfortability based on max_abs_lat_accel
        :param history: History from a simulation engine
        :param max_abs_lat_accel: Threshold for the absolute lat jerk
        :return True if within the threshold otherwise false.
        """
        ego_pose_states = history.extract_ego_state
        ego_pose_lat_accels = extract_ego_acceleration(ego_pose_states, acceleration_coordinate='y')

        lat_accels_within_bounds = np.abs(ego_pose_lat_accels) < max_abs_lat_accel

        return bool(np.all(lat_accels_within_bounds))

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the lateral acceleration metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated lateral acceleration metric.
        """
        metric_statistics: List[MetricStatistics] = self._compute_statistics(
            history=history,
            scenario=scenario,
            statistic_unit_name='meters_per_second_squared',
            extract_function=extract_ego_acceleration,
            extract_function_params={'acceleration_coordinate': 'y'},
            min_within_bound_threshold=-self._max_abs_lat_accel,
            max_within_bound_threshold=self._max_abs_lat_accel,
        )
        return metric_statistics
