from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import in_collision
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic
from nuplan.planning.metrics.utils.collision_utils import AtFaultCollision, CollisionType, ego_delta_v_collision
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind, is_track_stopped
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


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

    timestamp: int  # The timestamp at time of collision
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


def _classify_at_fault_collisions(
    all_collisions: List[Collisions],
    timestamps_in_common_or_connected_route_objs: List[int],
) -> Dict[TrackedObjectType, List[AtFaultCollision]]:
    """
    Returns a dictionary of track types and AtFaultCollision classes.

    We consider at_fault_collisions as collisions that could have been prevented if planner
    performed differently. For simplicity we call these collisions at fault although the proposed classification is
    not complete and there are more cases to be considered.

    :param all_collisions: List of all collisions in the history.
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
    lanes/lane connectors
    :return: A dict of at fault collisions and their track types.
    """
    at_fault_collisions: Dict[TrackedObjectType, List[AtFaultCollision]] = defaultdict(lambda: [])

    for collision in all_collisions:
        timestamp = collision.timestamp
        ego_in_multiple_lanes_or_nondrivable_area = timestamp not in timestamps_in_common_or_connected_route_objs

        for _id, collision_data in collision.collisions_id_data.items():
            # Include front collisions and collisions with stopped track in violation metric
            collisions_at_stopped_track_or_active_front = collision_data.collision_type in [
                CollisionType.ACTIVE_FRONT_COLLISION,
                CollisionType.STOPPED_TRACK_COLLISION,
            ]

            # Include lateral collisions if ego was in multiple lanes (e.g. during lane change) in violation metric
            collision_at_lateral = collision_data.collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

            if collisions_at_stopped_track_or_active_front or (
                ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
            ):

                at_fault_collisions[collision_data.tracked_object_type].append(
                    AtFaultCollision(
                        timestamp=timestamp,
                        duration=1,
                        collision_ego_delta_v=collision_data.collision_ego_delta_v,
                        collision_type=collision_data.collision_type,
                    )
                )

    return at_fault_collisions


class EgoAtFaultCollisionStatistics(MetricBase):
    """
    Statistics on number and energy of collisions of ego.
    A collision is defined as the event of ego intersecting another bounding box. If the same collision lasts for
    multiple frames, it still counts as a single one and the first frame is evaluated for the violation metric.
    """

    def __init__(self, name: str, category: str, ego_lane_change_metric: EgoLaneChangeStatistics) -> None:
        """
        Initializes the EgoAtFaultCollisionStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_lane_change_metric: Lane chang metric computed prior to calling the current metric
        """
        super().__init__(name=name, category=category)

        # Store results and all_collisions to re-use in high level metrics
        self.results: List[MetricStatistics] = []
        self.all_collisions: List[Collisions] = []
        self.all_at_fault_collisions: Dict[TrackedObjectType, List[AtFaultCollision]] = defaultdict(lambda: [])
        self.timestamps_at_fault_collisions: List[int] = []

        # Initialize ego_lane_change_metric
        self._ego_lane_change_metric = ego_lane_change_metric

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the collision metric.
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated collision energy and counts.
        """
        # Load pre-calculated results from ego_lane_change metric
        assert (
            self._ego_lane_change_metric.results
        ), "_ego_lane_change_metric metric must be run prior to calling {}".format(self.name)
        timestamps_in_common_or_connected_route_objs: List[
            int
        ] = self._ego_lane_change_metric.timestamps_in_common_or_connected_route_objs

        all_collisions: List[Collisions] = []
        collided_track_ids: Set[str] = set()

        for sample in history.data:
            ego_state = sample.ego_state
            observation = sample.observation
            timestamp = ego_state.time_point.time_us

            collided_track_ids, collisions_id_data = _find_new_collisions(ego_state, observation, collided_track_ids)

            # Update list of collisions
            if len(collisions_id_data):
                all_collisions.append(Collisions(timestamp, collisions_id_data))

        # Create a dict of at_fault collisions based on their track types
        all_at_fault_collisions = _classify_at_fault_collisions(
            all_collisions, timestamps_in_common_or_connected_route_objs
        )

        number_of_at_fault_collisions = sum(
            len(track_collisions) for track_collisions in all_at_fault_collisions.values()
        )

        timestamps_at_fault_collisions = [
            collision.timestamp
            for track_collisions in all_at_fault_collisions.values()
            for collision in track_collisions
        ]

        statistics = {
            MetricStatisticsType.COUNT: Statistic(
                name=f'number_of_{self.name}',
                unit='count',
                value=number_of_at_fault_collisions,
            ),
        }

        results = self._construct_metric_results(metric_statistics=statistics, time_series=None, scenario=scenario)

        # Save to re-use in high level metrics
        self.results = results
        self.all_collisions = all_collisions
        self.all_at_fault_collisions = all_at_fault_collisions
        self.timestamps_at_fault_collisions = timestamps_at_fault_collisions

        return results  # type: ignore
