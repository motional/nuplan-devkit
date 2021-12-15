from typing import List

import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point, get_ego_distance_to_goal
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoDistanceToGoalStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Ego distance metric.
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

        goal = history.mission_goal
        goal_point = Point2D(x=goal.x, y=goal.y)

        distances = get_ego_distance_to_goal(history, goal_point)
        statistics = {MetricStatisticsType.MAX: Statistic(name="ego_max_distance_from_goal", unit="meters",
                                                          value=np.amax(distances)),
                      MetricStatisticsType.MIN: Statistic(name="ego_min_distance_from_goal", unit="meters",
                                                          value=np.amin(distances)),
                      MetricStatisticsType.P90: Statistic(name="ego_p90_distance_from_goal", unit="meters",
                                                          value=np.percentile(np.abs(distances), 90)),  # type:ignore
                      }
        time_stamps = extract_ego_time_point(history)
        time_series = TimeSeries(unit='meters',
                                 time_stamps=list(time_stamps),
                                 values=distances)
        result = MetricStatistics(metric_computator=self.name,
                                  name="ego_distance_to_goal_statistics",
                                  statistics=statistics,
                                  time_series=time_series,
                                  metric_category=self.category)

        return [result]
