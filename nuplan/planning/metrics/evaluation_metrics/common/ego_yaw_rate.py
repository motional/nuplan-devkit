from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives, extract_ego_heading, \
    extract_ego_time_point
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoYawRateStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Ego yaw rate metric.
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
    def extract_ego_yaw_rate(history: SimulationHistory) -> npt.NDArray[np.float32]:
        """
        Extract ego yaw rate over time.
        :param history: History from a simulation engine.
        :return An array of heading (yaw) rate over time.
        """

        headings = extract_ego_heading(history)
        time_points = extract_ego_time_point(history)
        yaw_rate: npt.NDArray[np.float32] = approximate_derivatives(headings, time_points / 1e6)  # convert to seconds

        return yaw_rate

    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """

        yaw_rate = self.extract_ego_yaw_rate(history=history)
        statistics = {MetricStatisticsType.MAX: Statistic(name="ego_max_yaw_rate", unit="radians_per_second",
                                                          value=np.amax(yaw_rate)),
                      MetricStatisticsType.MIN: Statistic(name="ego_min_yaw_rate", unit="radians_per_second",
                                                          value=np.amin(yaw_rate)),
                      MetricStatisticsType.P90: Statistic(name="ego_p90_yaw_rate", unit="radians_per_second",
                                                          value=np.percentile(np.abs(yaw_rate), 90)),  # type:ignore
                      }
        time_stamps = extract_ego_time_point(history)
        time_series = TimeSeries(unit='radians_per_second',
                                 time_stamps=list(time_stamps),
                                 values=list(yaw_rate))
        result = MetricStatistics(metric_computator=self.name,
                                  name="ego_yaw_rate_statistics",
                                  statistics=statistics,
                                  time_series=time_series,
                                  metric_category=self.category)

        return [result]
