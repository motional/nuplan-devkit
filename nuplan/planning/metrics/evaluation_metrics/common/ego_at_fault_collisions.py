from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Set, Tuple

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import in_collision
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.evaluation_metrics.base.violation_metric_base import ViolationMetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricViolation
from nuplan.planning.metrics.utils.state_extractors import ego_delta_v_collision
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind, is_track_stopped
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


@dataclass
class CollisionData:
    """
    Class to retain information about a collision.
    """

    collision_ego_delta_v: float  # Energy in collision
    collision_type: CollisionType  # Type of collision
    tracked_object_type: TrackedObjectType  # Track type


@dataclass
class Collisions:
    """
    Class to retain information about the collisions at a particular timestamp.
    """

    timestamp: float  # The timestamp at time of collision
    # Contains the ids of the tracks in collision and their mapped collision data
    collisions_id_data: Dict[str, CollisionData]


def _get_collision_type(
    ego_state: EgoState, tracked_object: TrackedObject, stopped_speed_threshhold: float = 5e-03
) -> CollisionType:
    """
    Classify collision between ego and the track
    :param ego_state: Ego's state at the current timestamp
    :param tracked_object: Tracked object
    :param stopped_speed_threshhold: Threshhold for 0 speed due to noise
    :return Collision type.
    """
    is_ego_stopped = True if ego_state.dynamic_car_state.speed <= stopped_speed_threshhold else False

    # Collisions at zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif is_track_stopped(tracked_object):
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_state.rear_axle, tracked_object.box.center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front collision when both ego and track are not stopped
    elif is_agent_ahead(ego_state.rear_axle, tracked_object.box.center, 25):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type


def _find_new_collisions(
    ego_state: EgoState, observation: DetectionsTracks, collided_track_ids: Set[str]
) -> Tuple[Set[str], Dict[str, CollisionData]]:
    """
    Identify and classify new collisions in a given timestamp. We assume that ego can only collide with an agent
    once in the scenario. Collided tracks will be removed from metrics evaluation at future timestamps.
    :param ego_state: Ego's state at the current timestamp
    :param observation: DetectionsTracks at the current timestamp
    :param collided_track_ids: Set of all collisions happend before the current timestamp
    :return Updated set of collided track ids and a dict of new collided tracks and their CollisionData.
    """
    collisions_id_data: Dict[str, CollisionData] = {}

    for tracked_object in observation.tracked_objects:
        # Identify new collisions
        if tracked_object.track_token not in collided_track_ids and in_collision(
            ego_state.car_footprint.oriented_box, tracked_object.box
        ):

            # Update set of collided track ids
            collided_track_ids.add(tracked_object.track_token)
            # Calculate energy at the time of collision
            collision_delta_v = ego_delta_v_collision(ego_state, tracked_object)
            # Classify collision type
            collision_type = _get_collision_type(ego_state, tracked_object)

            collisions_id_data[tracked_object.track_token] = CollisionData(
                collision_delta_v, collision_type, tracked_object.tracked_object_type
            )

    return collided_track_ids, collisions_id_data


def _compute_violations(
    timestamp: int,
    collisions_id_data: Dict[str, CollisionData],
    timestamps_in_common_or_connected_route_objs: List[int],
    metric_name: str,
    metric_category: str,
) -> List[MetricViolation]:
    """
    Computes the violation metric for collisions at current timestamp.

    We consider the violation metric only for some specific collisions that could have been prevented if planner
    performed differently. For simplicity we call these collisions at fault although the proposed classification is
    not complete and there are more cases to be considered.

    :param timestamp: The current timestamp
    :param collisions_id_data: Dict of collision tracks and their CollisionData at current timestamp
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
    lanes/lane connectors
    :param metric_name: Metric name
    :param metric_category: Metric category
    :return: List of violations at the current timestamp.
    """
    violations: List[MetricViolation] = []
    ego_in_multiple_lanes = timestamp not in timestamps_in_common_or_connected_route_objs

    for _id, collisions_data in collisions_id_data.items():
        # Include front collisions and collisions with stopped track in violation metric
        collisions_at_stopped_track_or_active_front = collisions_data.collision_type in [
            CollisionType.ACTIVE_FRONT_COLLISION,
            CollisionType.STOPPED_TRACK_COLLISION,
        ]

        # Include lateral collisions if ego was in multiple lanes (e.g. during lane change) in violation metric
        collision_at_lateral = collisions_data.collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

        if collisions_at_stopped_track_or_active_front or (ego_in_multiple_lanes and collision_at_lateral):

            violations.append(
                MetricViolation(
                    metric_computator=metric_name,
                    name=metric_name,
                    metric_category=metric_category,
                    unit="meters per second",
                    start_timestamp=timestamp,
                    duration=1,
                    extremum=collisions_data.collision_ego_delta_v,
                    mean=collisions_data.collision_ego_delta_v,
                )
            )
    return violations


class EgoAtFaultCollisionStatistics(ViolationMetricBase):
    """
    Statistics on number and energy of collisions of ego.
    A collision is defined as the event of ego intersecting another bounding box. If the same collision lasts for
    multiple frames, it still counts as a single one and the first frame is evaluated for the violation metric.
    """

    def __init__(
        self, name: str, category: str, ego_lane_change_metric: EgoLaneChangeStatistics, max_violation_threshold: int
    ) -> None:
        """
        Initializes the EgoAtFaultCollisionStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_lane_change_metric: Lane chang metric computed prior to calling the current metric
        :param max_violation_threshold: Maximum threshold for the violation.
        """
        super().__init__(name=name, category=category, max_violation_threshold=max_violation_threshold)

        # Store results and all_collisions to re-use in high level metrics
        self.results: List[MetricStatistics] = []
        self.all_collisions: List[Collisions] = []
        self.all_violations: List[MetricViolation] = []

        # Initialize ego_lane_change_metric
        self._ego_lane_change_metric = ego_lane_change_metric

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the collision metric.
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated collision energy and counts.
        """
        timestamps_in_common_or_connected_route_objs: List[
            int
        ] = self._ego_lane_change_metric.timestamps_in_common_or_connected_route_objs

        all_violations: List[MetricViolation] = []
        all_collisions: List[Collisions] = []
        collided_track_ids: Set[str] = set()

        for sample in history.data:
            ego_state = sample.ego_state
            observation = sample.observation
            timestamp = ego_state.time_point.time_us

            collided_track_ids, collisions_id_data = _find_new_collisions(ego_state, observation, collided_track_ids)

            # Update list of collisions and violations
            if len(collisions_id_data):
                all_collisions.append(Collisions(timestamp, collisions_id_data))
                all_violations.extend(
                    _compute_violations(
                        timestamp,
                        collisions_id_data,
                        timestamps_in_common_or_connected_route_objs,
                        self.name,
                        self.category,
                    )
                )

        violation_statistics = self.aggregate_metric_violations(all_violations, scenario=scenario)

        # Save to re-use in high level metrics
        self.results = violation_statistics
        self.all_collisions = all_collisions
        self.all_violations = all_violations

        return violation_statistics  # type: ignore
