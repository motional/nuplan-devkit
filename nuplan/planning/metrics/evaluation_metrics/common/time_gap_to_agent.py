from typing import List

import numpy as np
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import get_closest_agent_in_position, \
    is_agent_ahead, is_agent_behind


def compute_time_gap(distance: float, velocity: float) -> float:
    return distance / (max(velocity, 0.1))  # clamp to avoid zero division


class TimeGapToLeadAgent(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Time Gap metric.
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

        time_gap = []
        for sample in history.data:
            ego_state = sample.ego_state
            observations = sample.observation
            closest_agent, closest_distance = get_closest_agent_in_position(ego_state, observations, is_agent_ahead)
            time_gap.append(compute_time_gap(closest_distance, ego_state.dynamic_car_state.speed))

        # Extract timestamps and remove nan values
        time_stamps = extract_ego_time_point(history)
        time_gap = np.asarray(time_gap)  # type: ignore
        time_gap = time_gap[np.isfinite(time_gap)]

        # If there are values, we compute statistics. Otherwise, assign nan to statistics
        if len(time_gap):
            max_value = np.amax(time_gap)
            min_value = np.amin(time_gap)
            percentile_90 = np.percentile(np.abs(time_gap), 90)  # type: ignore
        else:
            max_value = min_value = percentile_90 = np.nan

        # Insert nan values
        nan_values = [np.nan] * (len(time_stamps) - len(time_gap))
        time_gap = np.insert(time_gap, len(time_gap), nan_values)  # type: ignore
        statistics = {MetricStatisticsType.MAX: Statistic(name="max_time_gap_to_lead_agent", unit="seconds",
                                                          value=max_value),
                      MetricStatisticsType.MIN: Statistic(name="min_time_gap_to_lead_agent", unit="seconds",
                                                          value=min_value),
                      MetricStatisticsType.P90: Statistic(name="p90_time_gap_to_lead_agent", unit="seconds",
                                                          value=percentile_90),
                      }
        time_series = TimeSeries(unit='seconds',
                                 time_stamps=list(time_stamps),
                                 values=time_gap)
        result = MetricStatistics(metric_computator=self.name,
                                  name="time_gap_to_lead_agent",
                                  statistics=statistics,
                                  time_series=time_series, metric_category=self.category)

        return [result]


class TimeGapToRearAgent(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Time Gap metric.
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

        time_gap = []
        for sample in history.data:
            ego_state = sample.ego_state
            observations = sample.observation
            closest_agent, closest_distance = get_closest_agent_in_position(ego_state, observations, is_agent_behind)

            if closest_agent is not None:
                # get the rotation matrix of the agent to transform velocity to ego reference
                rotation_matrix = closest_agent.rotation_matrix
                agent_velocity_rotated = np.dot(rotation_matrix, closest_agent.velocity)  # type: ignore
                agent_velocity = np.hypot(agent_velocity_rotated[0], agent_velocity_rotated[1])

            else:
                agent_velocity = 0

            time_gap.append(compute_time_gap(closest_distance, agent_velocity))

        # Extract timestamps and remove nan values
        time_stamps = extract_ego_time_point(history)
        time_gap = np.asarray(time_gap)  # type: ignore
        time_gap = time_gap[np.isfinite(time_gap)]

        # If there are values, we compute statistics. Otherwise, assign nan to statistics
        if len(time_gap):
            max_value = np.amax(time_gap)
            min_value = np.amin(time_gap)
            percentile_90 = np.percentile(np.abs(time_gap), 90)  # type: ignore
        else:
            max_value = min_value = percentile_90 = np.nan

        # Insert nan values
        nan_values = [np.nan] * (len(time_stamps) - len(time_gap))
        time_gap = np.insert(time_gap, len(time_gap), nan_values)  # type: ignore
        statistics = {MetricStatisticsType.MAX: Statistic(name="max_time_gap_to_rear_agent", unit="seconds",
                                                          value=max_value),
                      MetricStatisticsType.MIN: Statistic(name="min_time_gap_to_rear_agent", unit="seconds",
                                                          value=min_value),
                      MetricStatisticsType.P90: Statistic(name="p90_time_gap_to_rear_agent", unit="seconds",
                                                          value=percentile_90),
                      }
        time_series = TimeSeries(unit='seconds',
                                 time_stamps=list(time_stamps),
                                 values=time_gap)
        result = MetricStatistics(metric_computator=self.name,
                                  name="time_gap_from_rear_agent",
                                  statistics=statistics,
                                  time_series=time_series,
                                  metric_category=self._category)
        return [result]
