from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils import state_extractors
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class TimeToCollisionStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str, collision_circle_radius: float, time_step_size: float,
                 time_horizon: float) -> None:
        """
        Ego time to collision metric, reports the minimal time for a projected collision if agents proceed with
        zero acceleration.

        :param name: Metric name.
        :param category: Metric category.
        :param collision_circle_radius: Radius of the circle used for collision checking [m]
        :param time_step_size: Step size for the propagation of collision circles
        :param time_horizon: Time horizon for collision checking
        """

        self._name = name
        self._category = category
        self._collision_circle_radius = collision_circle_radius
        self._time_step_size = time_step_size
        self._time_horizon = time_horizon

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

    def compute_time_to_collision(self, history: SimulationHistory) -> npt.NDArray[np.float32]:
        """
        Computes an estimate of the minimal time to collision with other agents. Agents are simplified with circles
        which are propagated in a straight line with constant velocity until there is a collision with ego or the
        maximal time window is reached.

        :param history The scenario history
        :return: The minimal TTC for each sample, inf if no collision is found
        """
        ego_states = [sample.ego_state for sample in history.data]
        ego_velocities = state_extractors.extract_ego_velocity(history)

        all_tracks_poses = state_extractors.extract_tracks_poses(history)
        all_tracks_speed = state_extractors.extract_tracks_speed(history)

        # Default to be inf
        time_to_collision: npt.NDArray[np.float32] = np.asarray([np.inf] * len(history))
        for i, (ego_state, ego_speed, tracks_poses, tracks_speed) in enumerate(zip(ego_states,
                                                                                   ego_velocities,
                                                                                   all_tracks_poses,
                                                                                   all_tracks_speed)):
            # Remain inf if we don't have any agents
            if len(tracks_poses) == 0:
                continue

            ego_dx = np.cos(ego_state.rear_axle.heading) * ego_speed * self._time_step_size
            ego_dy = np.sin(ego_state.rear_axle.heading) * ego_speed * self._time_step_size

            tracks_dxy = np.array([np.cos(tracks_poses[:, 2]) * tracks_speed * self._time_step_size,
                                   np.sin(tracks_poses[:, 2]) * tracks_speed * self._time_step_size]).T
            t = 0.0
            ego_pose = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y])
            while t < self._time_horizon:
                tracks_poses[:, :2] += tracks_dxy
                ego_pose += (ego_dx, ego_dy)
                dist = np.linalg.norm(tracks_poses[:, :2] - ego_pose, axis=1)

                if np.min(dist) < 2 * self._collision_circle_radius:
                    break

                t += self._time_step_size
            time_to_collision[i] = t if t < self._time_horizon else np.inf
        return time_to_collision

    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """
        time_to_collision = self.compute_time_to_collision(history=history)
        statistics = {MetricStatisticsType.MIN: Statistic(name="min_time_to_collision", unit="seconds",
                                                          value=np.amin(time_to_collision)),
                      MetricStatisticsType.P90: Statistic(name="ego_p90_time_to_collision", unit="seconds",
                                                          value=np.percentile(time_to_collision, 90,
                                                                              interpolation='nearest')),
                      }

        time_stamps = state_extractors.extract_ego_time_point(history)
        time_series = TimeSeries(unit='seconds',
                                 time_stamps=list(time_stamps),
                                 values=list(time_to_collision))
        result = MetricStatistics(metric_computator=self.name,
                                  name="time_to_collision_statistics",
                                  statistics=statistics,
                                  time_series=time_series,
                                  metric_category=self.category)

        return [result]
