from typing import List

import numpy as np
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_acceleration, extract_ego_jerk, \
    extract_ego_time_point
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoJerkStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Ego jerk metric.
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

    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """

        accelerations = extract_ego_acceleration(history)
        jerk = extract_ego_jerk(history, accelerations=accelerations)
        timestamps = extract_ego_time_point(history=history)

        statistics = {MetricStatisticsType.MAX: Statistic(name="ego_max_jerk", unit="meters_per_second_cubed",
                                                          value=np.amax(jerk)),
                      MetricStatisticsType.MIN: Statistic(name="ego_min_jerk", unit="meters_per_second_cubed",
                                                          value=np.amin(jerk)),
                      MetricStatisticsType.P90: Statistic(name="ego_p90_jerk", unit="meters_per_second_cubed",
                                                          value=np.percentile(np.abs(jerk), 90)),  # type:ignore
                      }

        time_series = TimeSeries(unit='meters_per_second_cubed',
                                 time_stamps=list(timestamps),
                                 values=list(jerk))
        result = MetricStatistics(metric_computator=self.name,
                                  name="ego_jerk_statistics",
                                  statistics=statistics,
                                  time_series=time_series, metric_category=self.category)
        return [result]
