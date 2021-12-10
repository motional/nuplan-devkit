from typing import List

import numpy as np
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.evaluation_metrics.common.ego_jerk import EgoJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_acceleration import EgoLatAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration import EgoLonAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_jerk import EgoLonJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_acceleration import EgoYawAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_rate import EgoYawRateStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoIsComfortableStatistics(AbstractMetricBuilder):

    def __init__(self,
                 name: str,
                 category: str,
                 min_lon_accel: float,
                 max_lon_accel: float,
                 max_abs_lat_accel: float,
                 max_abs_yaw_rate: float,
                 max_abs_yaw_accel: float,
                 max_abs_lon_jerk: float,
                 max_abs_mag_jerk: float,
                 ) -> None:
        """
        Check if ego trajectory is comfortable based on min_ego_lon_acceleration, max_ego_lon_acceleration,
        max_ego_abs_lat_acceleration, max_ego_abs_yaw_rate, max_ego_abs_yaw_acceleration, max_ego_abs_jerk_lon,
        max_ego_abs_jerk.
        :param name: Metric name.
        :param category: Metric category.
        :param min_lon_accel: Minimum longitudinal acceleration threshold.
        :param max_lon_accel: Maximum longitudinal acceleration threshold.
        :param max_abs_lat_accel: Maximum absolute lateral acceleration threshold.
        :param max_abs_yaw_rate: Maximum absolute yaw rate threshold.
        :param max_abs_yaw_accel: Maximum absolute yaw acceleration threshold.
        :param max_abs_lon_jerk: Maximum absolute longitudinal jerk threshold.
        :param max_abs_mag_jerk: Maximum absolute jerk magnitude threshold.
        """

        self._name = name
        self._category = category
        self._min_lon_accel = min_lon_accel
        self._max_lon_accel = max_lon_accel
        self._max_abs_lat_accel = max_abs_lat_accel
        self._max_abs_yaw_rate = max_abs_yaw_rate
        self._max_abs_yaw_accel = max_abs_yaw_accel
        self._max_abs_lon_jerk = max_abs_lon_jerk
        self._max_abs_mag_jerk = max_abs_mag_jerk

        self._ego_lon_accel = EgoLonAccelerationStatistics(name=self._name, category=self._category)
        self._ego_lat_accel = EgoLatAccelerationStatistics(name=self._name, category=self._category)
        self._ego_yaw_rate = EgoYawRateStatistics(name=self._name, category=self._category)
        self._ego_yaw_accel = EgoYawAccelerationStatistics(name=self._name, category=self._category)
        self._ego_lon_jerk = EgoLonJerkStatistics(name=self._name, category=self._category)
        self._ego_abs_jerk = EgoJerkStatistics(name=self._name, category=self._category)

    @property
    def name(self) -> str:
        """
        Returns the metric name.
        :return: the metric name.
        """

        return self._name

    @property
    def category(self) -> str:
        """
        Returns the metric category.
        :return: the metric category.
        """

        return self._category

    @staticmethod
    def _check_values_within_upper(metric_statistics: MetricStatistics, abs_upper_bound: float) -> bool:
        """
        Checks if the time series breaks the upper bound.

        :param metric_statistics: The metric statistic being checked
        :param abs_upper_bound: The upper bound (absolute value)
        :return: Whether the time series breaks the upper bound.
        """

        assert metric_statistics.time_series

        max_value = np.amax(np.abs(metric_statistics.time_series.values))
        return max_value < abs_upper_bound  # type: ignore

    def check_ego_is_comfortable(self, history: SimulationHistory) -> bool:
        """
        Check if ego trajectory is comfortable.
        :param history: History from a simulation engine.
        :return Ego comfortable status.
        """

        ego_lon_accel_result = self._ego_lon_accel.compute(history=history)[0]
        max_ego_lon_accel = ego_lon_accel_result.statistics[MetricStatisticsType.MAX].value
        min_ego_lon_accel = ego_lon_accel_result.statistics[MetricStatisticsType.MIN].value

        if min_ego_lon_accel < self._min_lon_accel or max_ego_lon_accel > self._max_lon_accel:
            return False

        if not self._check_values_within_upper(self._ego_lat_accel.compute(history=history)[0],
                                               self._max_abs_lat_accel):
            return False

        if not self._check_values_within_upper(self._ego_yaw_rate.compute(history=history)[0],
                                               self._max_abs_yaw_rate):
            return False

        if not self._check_values_within_upper(self._ego_yaw_accel.compute(history=history)[0],
                                               self._max_abs_yaw_accel):
            return False

        if not self._check_values_within_upper(self._ego_lon_jerk.compute(history=history)[0],
                                               self._max_abs_lon_jerk):
            return False

        if not self._check_values_within_upper(self._ego_abs_jerk.compute(history=history)[0],
                                               self._max_abs_mag_jerk):
            return False

        return True

    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """

        ego_is_comfortable = self.check_ego_is_comfortable(history=history)
        statistics = {
            MetricStatisticsType.BOOLEAN: Statistic(name="ego_is_comfortable", unit="boolean",
                                                    value=ego_is_comfortable)}
        result = MetricStatistics(metric_computator=self.name,
                                  name="ego_is_comfortable_statistics",
                                  statistics=statistics,
                                  time_series=None,
                                  metric_category=self.category)
        return [result]
