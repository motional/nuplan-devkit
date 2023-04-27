from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from shapely.geometry import LineString

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import in_collision
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.collision_utils import (
    CollisionType,
    ego_delta_v_collision,
    get_fault_type_statistics,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.utils import is_agent_behind, is_track_stopped
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
    ego_state: EgoState, tracked_object: TrackedObject, stopped_speed_threshold: float = 5e-02
) -> CollisionType:
    """
    Classify collision between ego and the track.
    :param ego_state: Ego's state at the current timestamp.
    :param tracked_object: Tracked object.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return Collision type.
    """
    is_ego_stopped = ego_state.dynamic_car_state.speed <= stopped_speed_threshold

    # Collisions at (close-to) zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif is_track_stopped(tracked_object):
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_state.rear_axle, tracked_object.box.center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front bumper collision when both ego and track are not stopped
    elif LineString(
        [
            ego_state.car_footprint.oriented_box.geometry.exterior.coords[0],
            ego_state.car_footprint.oriented_box.geometry.exterior.coords[3],
        ]
    ).intersects(tracked_object.box.geometry):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type


def find_new_collisions(
    ego_state: EgoState, observation: DetectionsTracks, collided_track_ids: Set[str]
) -> Tuple[Set[str], Dict[str, CollisionData]]:
    """
    Identify and classify new collisions in a given timestamp. We assume that ego can only collide with an agent
    once in the scenario. Collided tracks will be removed from metrics evaluation at future timestamps.
    :param ego_state: Ego's state at the current timestamp.
    :param observation: DetectionsTracks at the current timestamp.
    :param collided_track_ids: Set of all collisions happend before the current timestamp.
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


def classify_at_fault_collisions(
    all_collisions: List[Collisions],
    timestamps_in_common_or_connected_route_objs: List[int],
) -> Tuple[List[int], Dict[TrackedObjectType, List[float]]]:
    """
    Return a list of timestamps that at fault collisions happened and a dictionary of track types and collision energy.

    We consider at_fault_collisions as collisions that could have been prevented if planner
    performed differently. For simplicity we call these collisions at fault although the proposed classification is
    not complete and there are more cases to be considered.

    :param all_collisions: List of all collisions in the history.
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
    lanes/lane connectors.
    :return: A list of timestamps that at fault collisions happened and a dictionary of track types and collision energy.
    """
    at_fault_collisions: Dict[TrackedObjectType, List[float]] = defaultdict(list)
    timestamps_at_fault_collisions: List[int] = []
    for collision in all_collisions:
        timestamp = collision.timestamp
        ego_in_multiple_lanes_or_nondrivable_area = timestamp not in timestamps_in_common_or_connected_route_objs

        for _id, collision_data in collision.collisions_id_data.items():
            # Add front collisions and collisions with stopped track to at fault collisions
            collisions_at_stopped_track_or_active_front = collision_data.collision_type in [
                CollisionType.ACTIVE_FRONT_COLLISION,
                CollisionType.STOPPED_TRACK_COLLISION,
            ]

            # Add lateral collisions if ego was in multiple lanes (e.g. during lane change) to at fault collisions
            collision_at_lateral = collision_data.collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

            if collisions_at_stopped_track_or_active_front or (
                ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
            ):
                # Update the list of at fault collisions timestamps
                timestamps_at_fault_collisions.append(timestamp)
                # Update the list of collision energies for the collided track type
                at_fault_collisions[collision_data.tracked_object_type].append(collision_data.collision_ego_delta_v)

    return timestamps_at_fault_collisions, at_fault_collisions


class EgoAtFaultCollisionStatistics(MetricBase):
    """
    Statistics on number and energy of collisions of ego.
    A collision is defined as the event of ego intersecting another bounding box. If the same collision lasts for
    multiple frames, it still counts as a single one.
    """

    def __init__(
        self,
        name: str,
        category: str,
        ego_lane_change_metric: EgoLaneChangeStatistics,
        max_violation_threshold_vru: int = 0,
        max_violation_threshold_vehicle: int = 0,
        max_violation_threshold_object: int = 1,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initialize the EgoAtFaultCollisionStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param ego_lane_change_metric: Lane change metric computed prior to calling the current metric.
        :param max_violation_threshold_vru: Maximum threshold for the collision with VRUs.
        :param max_violation_threshold_vehicle: Maximum threshold for the collision with vehicles.
        :param max_violation_threshold_object: Maximum threshold for the collision with objects.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._max_violation_threshold_vru = max_violation_threshold_vru
        self._max_violation_threshold_vehicle = max_violation_threshold_vehicle
        self._max_violation_threshold_object = max_violation_threshold_object

        # Store results and all_collisions to re-use in high level metrics
        self.results: List[MetricStatistics] = []
        self.all_collisions: List[Collisions] = []
        self.all_at_fault_collisions: Dict[TrackedObjectType, List[float]] = defaultdict(list)
        self.timestamps_at_fault_collisions: List[int] = []

        # Initialize ego_lane_change_metric
        self._ego_lane_change_metric = ego_lane_change_metric

    def _compute_collision_score(self, number_of_collisions: int, max_violation_threshold: int) -> float:
        """
        Compute a score based on a maximum violation threshold. The score is max( 0, 1 - (x / (max_violation_threshold + 1)))
        The score will be 0 if the number of collisions exceeds this value.
        :param max_violation_threshold: Total number of allowed collisions.
        :return A metric score between 0 and 1.
        """
        return max(0.0, 1.0 - (number_of_collisions / (max_violation_threshold + 1)))

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> Optional[float]:
        """Inherited, see superclass.
        The total score for this metric is defined as the product of the scores for VRUs, vehicles and object track types. If no at fault collision exist, the score is 1.
        """
        return (
            1
            if metric_statistics[0].value  # no at fault collisions
            else self._compute_collision_score(
                metric_statistics[2].value, self._max_violation_threshold_vru
            )  # Collision score based on the number of at fault collisions with VRUs
            * self._compute_collision_score(
                metric_statistics[3].value, self._max_violation_threshold_vehicle
            )  # Collision score based on the number of at fault collisions with vehicles
            * self._compute_collision_score(
                metric_statistics[4].value, self._max_violation_threshold_object
            )  # Collision score based on the number of at fault collisions with objects
        )

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the collision metric.
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated collision energy and counts.
        """
        # Load pre-calculated results from ego_lane_change metric
        assert self._ego_lane_change_metric.results, "ego_lane_change_metric must be run prior to calling {}".format(
            self.name
        )
        timestamps_in_common_or_connected_route_objs: List[
            int
        ] = self._ego_lane_change_metric.timestamps_in_common_or_connected_route_objs

        all_collisions: List[Collisions] = []
        collided_track_ids: Set[str] = set()

        for sample in history.data:
            ego_state = sample.ego_state
            observation = sample.observation
            timestamp = ego_state.time_point.time_us

            collided_track_ids, collisions_id_data = find_new_collisions(ego_state, observation, collided_track_ids)

            # Update list of collisions
            if len(collisions_id_data):
                all_collisions.append(Collisions(timestamp, collisions_id_data))

        # Save at fault collisions timestamps and a dict of collision energies based on the track types
        self.timestamps_at_fault_collisions, self.all_at_fault_collisions = classify_at_fault_collisions(
            all_collisions, timestamps_in_common_or_connected_route_objs
        )

        number_of_at_fault_collisions = sum(
            len(track_collisions) for track_collisions in self.all_at_fault_collisions.values()
        )

        statistics = [
            Statistic(
                name=f"{self.name}",
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=number_of_at_fault_collisions == 0,
                type=MetricStatisticsType.BOOLEAN,
            ),
            Statistic(
                name='number_of_all_at_fault_collisions',
                unit=MetricStatisticsType.COUNT.unit,
                value=number_of_at_fault_collisions,
                type=MetricStatisticsType.COUNT,
            ),
        ]
        statistics.extend(get_fault_type_statistics(self.all_at_fault_collisions))

        # Save to re-use in high level metrics
        self.results = self._construct_metric_results(
            metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit
        )
        self.all_collisions = all_collisions

        return self.results
