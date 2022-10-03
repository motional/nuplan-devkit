from typing import List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox, in_collision
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
    Collisions,
    EgoAtFaultCollisionStatistics,
)
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point, extract_ego_velocity
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
from nuplan.planning.simulation.observation.observation_type import Observation

# Typing for trajectory of tracks pose, speed and box
TRACKS_POSE_SPEED_BOX = Tuple[
    List[npt.NDArray[np.float32]], List[npt.NDArray[np.float32]], List[npt.NDArray[OrientedBox]]
]


def extract_tracks_info_excluding_collided_tracks(
    ego_states: List[EgoState],
    ego_timestamps: npt.NDArray[np.int32],
    observations: List[Observation],
    all_collisions: List[Collisions],
    timestamps_in_common_or_connected_route_objs: List[int],
    map_api: AbstractMap,
) -> TRACKS_POSE_SPEED_BOX:
    """
    Extracts arrays of tracks pose, speed and oriented box for TTC: all lead and cross tracks, plus lateral tracks if ego is in
    between lanes or in nondrivable area or in intersection.

    :param ego_states: A list of ego states
    :param ego_timestamps: Array of times in time_us
    :param observations: A list of observations
    :param all_collisions: List of all collisions in the history
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
        lanes/lane connectors
    :param map_api: map api.
    :return: A tuple of lists of arrays of tracks pose, speed and represented box at each timestep.
    """
    collided_track_ids: Set[str] = set()

    history_tracks_poses: List[npt.NDArray[np.float32]] = []
    history_tracks_speed: List[npt.NDArray[np.float32]] = []
    history_tracks_boxes: List[npt.NDArray[OrientedBox]] = []

    collision_time_dict = {
        collision.timestamp: list(collision.collisions_id_data.keys()) for collision in all_collisions
    }

    for (ego_state, timestamp, observation) in zip(ego_states, ego_timestamps, observations):

        # Add the collided tracks at this timestamp to be excluded from later timestamps
        collided_track_ids = collided_track_ids.union(set(collision_time_dict.get(timestamp, [])))

        # Check if ego is in between lanes or between/in nondrivable area, in this case we consider
        # lead/cross and lateral tracks for TTC, otherwise only lead/cross tracks are considered in TTC.
        ego_not_in_common_or_connected_route_objs = timestamp not in timestamps_in_common_or_connected_route_objs

        tracked_objects = [
            tracked_object
            for tracked_object in observation.tracked_objects
            if tracked_object.track_token not in collided_track_ids
            and (
                is_agent_ahead(ego_state.rear_axle, tracked_object.center)
                or (
                    (
                        ego_not_in_common_or_connected_route_objs
                        or map_api.is_in_layer(ego_state.rear_axle, layer=SemanticMapLayer.INTERSECTION)
                    )
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
        history_tracks_boxes.append(np.array(boxes))

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
        no_ego_at_fault_collisions_metric: EgoAtFaultCollisionStatistics,
        time_step_size: float,
        time_horizon: float,
        least_min_ttc: float,
        metric_score_unit: Optional[str] = None,
    ):
        """
        Initializes the TimeToCollisionStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_lane_change_metric: Lane chang metric computed prior to calling the current metric
        :param no_ego_at_fault_collisions_metric: Ego at fault collisions computed prior to the current metric
        :param time_step_size: Step size for the propagation of collision agents
        :param time_horizon: Time horizon for collision checking
        :param least_min_ttc: minimum desired TTC.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)

        self._time_step_size = time_step_size
        self._time_horizon = time_horizon
        self._least_min_ttc = least_min_ttc

        # Initialize lower level metrics
        self._ego_lane_change_metric = ego_lane_change_metric
        self._no_ego_at_fault_collisions_metric = no_ego_at_fault_collisions_metric

        # save to load in higher level metrics
        self.results: List[MetricStatistics] = []

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        # Return 1.0 if time_to_collision_bound is True, otherwise 0
        return float(metric_statistics[-1].value)

    @staticmethod
    def _get_elongated_box_length(
        length: float, dx: float, dy: float, time_horizon: float, time_step_size: float
    ) -> float:
        """
        Helper to find the length of an elongated box projected up to a given time horizon.
        :param length: The length of the OrientedBox
        :param dx: Movement in x axis in global frame at each time_step_size
        :param dy: Movement in y axis in global frame at each time_step_size
        :param time_horizon: Time horizon for collision checking
        :param time_step_size: Step size for the propagation of collision agents
        :return: Length of elonated box up to time horizon.
        """
        return float(length + np.hypot(dx * time_horizon / time_step_size, dy * time_horizon / time_step_size))

    def compute_time_to_collision(
        self,
        history: SimulationHistory,
        ego_states: List[EgoState],
        ego_timestamps: npt.NDArray[np.int32],
        timestamps_in_common_or_connected_route_objs: List[int],
        all_collisions: List[Collisions],
        timestamps_at_fault_collisions: List[int],
        stopped_speed_threshold: float = 5e-03,
    ) -> npt.NDArray[np.float32]:
        """
        Computes an estimate of the minimal time to collision with other agents. Ego and agents are projected
        with constant velocity until there is a collision or the maximal time window is reached.

        :param history The scenario history
        :param ego_states: A list of ego states
        :param ego_timestamps: Array of times in time_us
        :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
        lanes/lane connectors
        :param all_collisions: List of all collisions in the history
        :param timestamps_at_fault_collisions: List of timestamps corresponding to at-fault-collisions in the history
        :param stopped_speed_threshold: Threshold for 0 speed due to noise
        :return: The minimal TTC for each sample, inf if no collision is found within the projection horizon.
        """
        # Extract speed of ego from history.
        ego_velocities = extract_ego_velocity(history)

        # Extract observation from history.
        observations = [sample.observation for sample in history.data]

        # Extract tracks info, collided tracks are removed from the analysis after the first timestamp of the collision
        (
            history_tracks_poses,
            history_tracks_speed,
            history_tracks_boxes,
        ) = extract_tracks_info_excluding_collided_tracks(
            ego_states,
            ego_timestamps,
            observations,
            all_collisions,
            timestamps_in_common_or_connected_route_objs,
            history.map_api,
        )

        # Default TTC to be inf
        time_to_collision: npt.NDArray[np.float32] = np.asarray([np.inf] * len(history))

        time_step_size = self._time_step_size
        time_horizon = self._time_horizon

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

            ego_in_at_fault_collision = timestamp in timestamps_at_fault_collisions
            # Set TTC to 0 if ego is in an at-fault collision
            if ego_in_at_fault_collision:
                time_to_collision[i] = 0
                continue

            # Remain inf if we don't have any agents or ego is stopped
            if len(tracks_poses) == 0 or ego_speed <= stopped_speed_threshold:
                continue

            ego_pose: npt.NDArray[np.float32] = np.array([*ego_state.center])
            ego_box = ego_state.car_footprint.oriented_box

            # Find ego movements in the global frame
            ego_dx = np.cos(ego_pose[2]) * ego_speed * time_step_size
            ego_dy = np.sin(ego_pose[2]) * ego_speed * time_step_size

            # Find tracks' movements in the global frame, assume all tracks also follow the bicycle dynamic model
            tracks_dxy = np.array(
                [
                    np.cos(tracks_poses[:, 2]) * tracks_speed * time_step_size,  # type:ignore
                    np.sin(tracks_poses[:, 2]) * tracks_speed * time_step_size,
                ]
            ).T

            # Find the center of elongated boxes if ego and tracks continue their movement with
            # the same speed and heading
            ego_elongated_box_center_pose: npt.NDArray[np.float32] = np.array(
                [
                    (time_horizon / time_step_size) / 2 * ego_dx + ego_pose[0],
                    (time_horizon / time_step_size) / 2 * ego_dy + ego_pose[1],
                    ego_pose[2],
                ]
            )

            ego_elongated_box = OrientedBox(
                StateSE2(*ego_elongated_box_center_pose),
                self._get_elongated_box_length(ego_box.length, ego_dx, ego_dy, time_horizon, time_step_size),
                ego_box.width,
                ego_box.height,
            )

            # Project tracks poses up to the time_horizon
            tracks_elongated_box_center_poses: npt.NDArray[np.float32] = np.concatenate(
                (
                    (time_horizon / time_step_size) / 2 * tracks_dxy + tracks_poses[:, :2],
                    tracks_poses[:, 2].reshape(-1, 1),
                ),
                axis=1,
            )
            # Find the convex hulls including tracks initial and projected corners
            tracks_elongated_boxes = [
                OrientedBox(
                    StateSE2(*track_elongated_box_center_pose),
                    self._get_elongated_box_length(
                        track_box.length, track_dxy[0], track_dxy[1], time_horizon, time_step_size
                    ),
                    track_box.width,
                    track_box.height,
                )
                for track_box, track_dxy, track_elongated_box_center_pose in zip(
                    tracks_boxes, tracks_dxy, tracks_elongated_box_center_poses
                )
            ]

            # Find relevant tracks for which the elongated box overlaps with ego elongated box
            relevant_tracks_mask = np.where(
                [in_collision(ego_elongated_box, track_elongated_box) for track_elongated_box in tracks_elongated_boxes]
            )[0]

            # If there is no relevant track affecting TTC, remain inf
            if not len(relevant_tracks_mask):
                continue

            # Find TTC for relevant tracks by projecting ego and tracks boxes with time_step_size
            ttc_found = False
            for t in np.arange(time_step_size, time_horizon, time_step_size):
                # project ego's center pose and footprint with a fixed speed
                ego_pose[:2] += (ego_dx, ego_dy)
                projected_ego_box = OrientedBox.from_new_pose(ego_box, StateSE2(*ego_pose))
                # project tracks's center pose and footprint with a fixed speed
                tracks_poses[:, :2] += tracks_dxy
                for track_box, track_pose in zip(
                    tracks_boxes[relevant_tracks_mask], tracks_poses[relevant_tracks_mask]
                ):
                    projected_track_box = OrientedBox.from_new_pose(track_box, StateSE2(*track_pose))
                    if in_collision(projected_ego_box, projected_track_box):
                        time_to_collision[i] = t
                        ttc_found = True
                        break

                if ttc_found:
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

        # Load pre-calculated results from ego_at_fault_collision metric
        assert (
            self._no_ego_at_fault_collisions_metric.results
        ), "no_ego_at_fault_collisions metric must be run prior to calling {}".format(self.name)

        all_collisions = self._no_ego_at_fault_collisions_metric.all_collisions
        timestamps_at_fault_collisions = self._no_ego_at_fault_collisions_metric.timestamps_at_fault_collisions

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
            timestamps_at_fault_collisions,
        )

        time_to_collision_within_bounds = self._least_min_ttc < np.array(time_to_collision)
        time_series = TimeSeries(
            unit='time_to_collision_under_' + f'{self._time_horizon}' + '_seconds [s]',
            time_stamps=list(ego_timestamps),
            values=list(time_to_collision),
        )
        metric_statistics = [
            Statistic(
                name='min_time_to_collision',
                unit='seconds',
                value=np.min(time_to_collision),
                type=MetricStatisticsType.MIN,
            ),
            Statistic(
                name=f'{self.name}',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=bool(np.all(time_to_collision_within_bounds)),
                type=MetricStatisticsType.BOOLEAN,
            ),
        ]
        # Save to load in high level metrics
        self.results = self._construct_metric_results(
            metric_statistics=metric_statistics,
            time_series=time_series,
            scenario=scenario,
            metric_score_unit=self.metric_score_unit,
        )

        return self.results
