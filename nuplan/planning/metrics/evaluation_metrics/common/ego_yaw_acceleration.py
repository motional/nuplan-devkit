from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives, extract_ego_heading, \
    extract_ego_time_point
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoYawAccelerationStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Ego yaw acceleration metric.
        :param name: Metric name.
        :param category: Metric category.
        """

        self._name = name
        self._category = category

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
    def extract_ego_yaw_acceleration(history: SimulationHistory) -> npt.NDArray[np.float32]:
        """
        Extract ego yaw acceleration over time.
        :param history: History from a simulation engine.
        :return An array of heading (yaw) acceleration over time.
        """

        headings = extract_ego_heading(history)
        time_points = extract_ego_time_point(history)
        yaw_accels: npt.NDArray[np.float32] = approximate_derivatives(headings, time_points / 1e6,
                                                                      deriv_order=2, poly_order=3)  # convert to seconds

        return yaw_accels

    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """

        yaw_accels = self.extract_ego_yaw_acceleration(history=history)
        statistics = {MetricStatisticsType.MAX: Statistic(name="ego_max_yaw_acceleration", unit="radians_per_second",
                                                          value=np.amax(yaw_accels)),
                      MetricStatisticsType.MIN: Statistic(name="ego_min_yaw_acceleration", unit="radians_per_second",
                                                          value=np.amin(yaw_accels)),
                      MetricStatisticsType.P90: Statistic(name="ego_p90_yaw_acceleration", unit="radians_per_second",
                                                          value=np.percentile(np.abs(yaw_accels), 90))  # type:ignore
                      }
        time_stamps = extract_ego_time_point(history)
        time_series = TimeSeries(unit='radians_per_second',
                                 time_stamps=list(time_stamps),
                                 values=list(yaw_accels))
        result = MetricStatistics(metric_computator=self.name,
                                  name="ego_yaw_acceleration_statistics",
                                  statistics=statistics,
                                  time_series=time_series,
                                  metric_category=self.category)

        return [result]
