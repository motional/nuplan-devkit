from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox, in_collision
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_at_fault_collisions import (
    Collisions,
    EgoAtFaultCollisionStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.metric_result import (
    MetricStatistics,
    MetricStatisticsType,
    MetricViolation,
    Statistic,
    TimeSeries,
)
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point, extract_ego_velocity
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind

TRACKS_POSE_SPEED_BOX = Tuple[List[npt.NDArray[np.float32]], List[npt.NDArray[np.float32]], List[List[OrientedBox]]]


def extract_tracks_info_excluding_collided_tracks(
    history: SimulationHistory,
    all_collisions: List[Collisions],
    timestamps_in_common_or_connected_route_objs: List[int],
) -> TRACKS_POSE_SPEED_BOX:
    """
    Extracts tracks pose, speed and oriented box for TTC: all lead tracks, plus lateral tracks if ego is in between lanes
    :param history: History from a simulation engine
    :param all_collisions: List of all collisions in the history
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
        lanes/lane connectors
    :return: A tuple of lists of arrays containing tracks poses, speed and represented box at each timestep.
    """
    collided_track_ids: Set[str] = set()

    timestamps_in_collision = [collision.timestamp for collision in all_collisions]

    history_tracks_poses: List[npt.NDArray[np.float32]] = []
    history_tracks_speed: List[npt.NDArray[np.float32]] = []
    history_tracks_boxes: List[List[OrientedBox]] = []

    for sample in history.data:
        ego_state = sample.ego_state
        timestamp = ego_state.time_point.time_us
        observation = sample.observation

        # Check if ego is in a collision
        ego_in_collision = timestamp in timestamps_in_collision
        # Check if ego is in between lanes or between drivable and nondrivable area, in this case we consider all tracks (lead, lateral and lag) for TTC
        ego_not_in_common_or_connected_route_objs = timestamp not in timestamps_in_common_or_connected_route_objs

        if ego_in_collision:
            new_collided_track = [
                list(collision.collisions_id_data.keys())
                for collision in all_collisions
                if collision.timestamp == timestamp
            ][0]
            collided_track_ids = collided_track_ids.union(set(new_collided_track))

        tracked_objects = [
            tracked_object
            for tracked_object in observation.tracked_objects
            if tracked_object.track_token not in collided_track_ids
            and (
                is_agent_ahead(ego_state.rear_axle, tracked_object.center)
                or (
                    ego_not_in_common_or_connected_route_objs
                    and not is_agent_behind(ego_state.rear_axle, tracked_object.center)
                )
            )
        ]

        poses: List[npt.NDArray[np.float32]] = [
            np.array([*tracked_object.center]) for tracked_object in tracked_objects
        ]

        speeds: List[npt.NDArray[np.float32]] = [
            np.array(tracked_object.velocity.magnitude()) if isinstance(tracked_object, Agent) else 0  # type: ignore
            for tracked_object in tracked_objects
        ]

        boxes: List[OrientedBox] = [tracked_object.box for tracked_object in tracked_objects]

        history_tracks_poses.append(np.array(poses))
        history_tracks_speed.append(np.array(speeds))
        history_tracks_boxes.append(boxes)

    return history_tracks_poses, history_tracks_speed, history_tracks_boxes


class TimeToCollisionStatistics(MetricBase):
    """
    Ego time to collision metric, reports the minimal time for a projected collision if agents proceed with
    zero acceleration.
    """

    def __init__(
        self,
        name: str,
        category: str,
        ego_lane_change_metric: EgoLaneChangeStatistics,
        ego_at_fault_collisions_metric: EgoAtFaultCollisionStatistics,
        time_step_size: float,
        time_horizon: float,
        least_min_ttc: float,
    ):
        """
        Initializes the TimeToCollisionStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_lane_change_metric: Lane chang metric computed prior to calling the current metric
        :param ego_at_fault_collisions_metric: Ego at fault collisions metric computed prior to the current metric
        :param time_step_size: Step size for the propagation of collision agents
        :param time_horizon: Time horizon for collision checking
        :param least_min_ttc: minimum desired TTC.
        """
        super().__init__(name=name, category=category)

        self._time_step_size = time_step_size
        self._time_horizon = time_horizon
        self._least_min_ttc = least_min_ttc

        # Initialize lower level metrics
        self._ego_lane_change_metric = ego_lane_change_metric
        self._ego_at_fault_collisions_metric = ego_at_fault_collisions_metric

        # save to load in higher level metrics
        self.results: List[MetricStatistics] = []

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        # Return 1.0 if time_to_collision_bound is True, otherwise 0
        return float(metric_statistics[MetricStatisticsType.BOOLEAN].value)

    def compute_time_to_collision(
        self,
        history: SimulationHistory,
        ego_states: List[EgoState],
        ego_timestamps: List[int],
        timestamps_in_common_or_connected_route_objs: List[int],
        all_collisions: List[Collisions],
        all_violations: List[MetricViolation],
        stopped_speed_threshhold: float = 5e-03,
    ) -> npt.NDArray[np.float32]:
        """
        Computes an estimate of the minimal time to collision with other agents. Agents are projected
        with constant velocity until there is a collision with ego or the maximal time window is reached.

        :param history The scenario history
        :param ego_states: A list of ego states
        :param ego_timestamps: Array of times in time_us
        :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
        lanes/lane connectors
        :param all_collisions: List of all collisions in the history
        :param all_violations: List of violations corresponding to at-fault-collisions in the history
        :param stopped_speed_threshhold: Threshhold for 0 speed due to noise
        :return: The minimal TTC for each sample, inf if no collision is found within the projection horizon.
        """
        # Extract speed of ego from history.
        ego_velocities = extract_ego_velocity(history)

        # Default TTC to be inf
        time_to_collision: npt.NDArray[np.float32] = np.asarray([np.inf] * len(history))

        timestamps_in_at_fault_collisions = [violation.start_timestamp for violation in all_violations]

        # Extract tracks info, collided tracks ar removed from the analysis after the first timestamp of the collision
        (
            history_tracks_poses,
            history_tracks_speed,
            history_tracks_boxes,
        ) = extract_tracks_info_excluding_collided_tracks(
            history, all_collisions, timestamps_in_common_or_connected_route_objs
        )

        for i, (timestamp, ego_state, ego_speed, tracks_poses, tracks_speed, tracks_boxes) in enumerate(
            zip(
                ego_timestamps,
                ego_states,
                ego_velocities,
                history_tracks_poses,
                history_tracks_speed,
                history_tracks_boxes,
            )
        ):

            ego_in_at_fault_collision = timestamp in timestamps_in_at_fault_collisions
            # Set TTC to 0 if ego is in an at-fault collision
            if ego_in_at_fault_collision:
                time_to_collision[i] = 0
                continue

            # Remain inf if we don't have any agents or ego is stopped
            if len(tracks_poses) == 0 or ego_speed <= stopped_speed_threshhold:
                continue

            ego_dx = np.cos(ego_state.center.heading) * ego_speed * self._time_step_size
            ego_dy = np.sin(ego_state.center.heading) * ego_speed * self._time_step_size

            tracks_dxy = np.array(
                [
                    np.cos(tracks_poses[:, 2]) * tracks_speed * self._time_step_size,  # type:ignore
                    np.sin(tracks_poses[:, 2]) * tracks_speed * self._time_step_size,
                ]
            ).T

            ego_pose: npt.NDArray[np.float32] = np.array([*ego_state.center])
            ego_box = ego_state.car_footprint.oriented_box

            # If there is no collision at this timestamp, find TTC by projecting oriented boxes with fixed speed and headings
            for t in np.arange(self._time_step_size, self._time_horizon, self._time_step_size):
                # project ego's center pose and footprint with a fixed speed
                ego_pose[:2] += (ego_dx, ego_dy)
                ego_box = OrientedBox.from_new_pose(ego_box, StateSE2.deserialize(ego_pose))
                # project tracks's center pose and footprint with a fixed speed
                tracks_poses[:, :2] += tracks_dxy
                tracks_boxes = [
                    OrientedBox.from_new_pose(track_box, StateSE2.deserialize(track_pose))
                    for track_box, track_pose in zip(tracks_boxes, tracks_poses)
                ]
                if np.any([in_collision(ego_box, track_box) for track_box in tracks_boxes]):
                    time_to_collision[i] = t
                    break

        return time_to_collision

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the time to collision statistics
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the time to collision metric
        """
        # Load pre-calculated timestamps from ego_lane_change_metric
        timestamps_in_common_or_connected_route_objs: List[
            int
        ] = self._ego_lane_change_metric.timestamps_in_common_or_connected_route_objs

        # Load pre-calculated collisions and violations from ego_at_fault_collision metric
        all_collisions = self._ego_at_fault_collisions_metric.all_collisions
        all_violations = self._ego_at_fault_collisions_metric.all_violations

        # Extract states of ego from history.
        ego_states = history.extract_ego_state

        # Extract ego timepoints
        ego_timestamps = extract_ego_time_point(ego_states)

        time_to_collision = self.compute_time_to_collision(
            history,
            ego_states,
            ego_timestamps,
            timestamps_in_common_or_connected_route_objs,
            all_collisions,
            all_violations,
        )

        time_series = TimeSeries(unit='seconds', time_stamps=list(ego_timestamps), values=list(time_to_collision))

        metric_statistics = self._compute_time_series_statistic(time_series=time_series)

        time_to_collision_within_bounds = self._least_min_ttc < np.array(time_to_collision)
        metric_statistics[MetricStatisticsType.BOOLEAN] = Statistic(
            name=f'{self.name}_within_bound', unit='boolean', value=bool(np.all(time_to_collision_within_bounds))
        )
        results = self._construct_metric_results(
            metric_statistics=metric_statistics, time_series=time_series, scenario=scenario
        )
        # Save to load in high level metrics
        self.results = results

        return results  # type: ignore
