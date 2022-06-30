from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.database.nuplan_db.prediction_construction import get_interpolated_waypoints
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)

LIDAR_PC_CACHE = 16 * 2**10  # 16K

DEFAULT_SCENARIO_NAME = 'unknown'  # name of scenario (e.g. ego overtaking)
DEFAULT_SCENARIO_DURATION = 20.0  # [s] duration of the scenario (e.g. extract 20s from when the event occurred)
DEFAULT_EXTRACTION_OFFSET = 0.0  # [s] offset of the scenario (e.g. start at -5s from when the event occurred)
DEFAULT_SUBSAMPLE_RATIO = 1.0  # ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)


@dataclass(frozen=True)
class ScenarioExtractionInfo:
    """
    Structure containing information used to extract a scenario (lidarpc sequence).
    """

    scenario_name: str = DEFAULT_SCENARIO_NAME  # name of the scenario
    scenario_duration: float = DEFAULT_SCENARIO_DURATION  # [s] duration of the scenario
    extraction_offset: float = DEFAULT_EXTRACTION_OFFSET  # [s] offset of the scenario
    subsample_ratio: float = DEFAULT_SUBSAMPLE_RATIO  # ratio to sample the scenario

    def __post_init__(self) -> None:
        """Sanitize class attributes."""
        assert 0.0 < self.scenario_duration, f'Scenario duration has to be greater than 0, got {self.scenario_duration}'
        assert (
            0.0 < self.subsample_ratio <= 1.0
        ), f'Subsample ratio has to be between 0 and 1, got {self.subsample_ratio}'


class ScenarioMapping:
    """
    Structure that maps each scenario type to instructions used in extracting it.
    """

    def __init__(self, scenario_map: Dict[str, Tuple[float, float, float]]) -> None:
        """
        Initializes the scenario mapping class.
        :param scenario_map: Dictionary with scenario name/type as keys and
                             tuples of (scenario duration, extraction offset, subsample ratio) as values.
        """
        self.mapping = {name: ScenarioExtractionInfo(name, *value) for name, value in scenario_map.items()}

    def get_extraction_info(self, scenario_type: str) -> Optional[ScenarioExtractionInfo]:
        """
        Accesses the scenario mapping using a query scenario type.
        If the scenario type is not found, a default extraction info object is returned.
        :param scenario_type: Scenario type to query for.
        :return: Scenario extraction information for the queried scenario type.
        """
        return self.mapping[scenario_type] if scenario_type in self.mapping else ScenarioExtractionInfo()


def get_time_stamp_from_lidar_pc(lidar_pc: LidarPc) -> int:
    """
    Extracts the time stamp from a LidarPc.
    :param lidar_pc: Input lidar pc.
    :return: Timestamp in micro seconds.
    """
    return cast(int, lidar_pc.ego_pose.timestamp)


def lidarpc_to_state_se2(lidar_pc: LidarPc) -> StateSE2:
    """
    Converts a LidarPc to an StateSE2 object.
    :param lidar_pc: Input lidar pc.
    :return: Instantiated state SE2.
    """
    ego_pose = lidar_pc.ego_pose
    return StateSE2(ego_pose.x, ego_pose.y, ego_pose.quaternion.yaw_pitch_roll[0])


# Do not cache this method.
# This will keep old versions of the DB alive, which will lead to excessive memory allocations.
def lidarpc_to_ego_state(lidar_pc: LidarPc) -> EgoState:
    """
    Converts a LidarPc to an EgoState object.
    :param lidar_pc: Input lidar pc.
    :return: Instantiated ego state.
    """
    ego_pose = lidar_pc.ego_pose
    return EgoState.build_from_rear_axle(
        StateSE2(ego_pose.x, ego_pose.y, ego_pose.quaternion.yaw_pitch_roll[0]),
        tire_steering_angle=0.0,
        vehicle_parameters=get_pacifica_parameters(),
        time_point=TimePoint(ego_pose.timestamp),
        rear_axle_velocity_2d=StateVector2D(x=ego_pose.vx, y=ego_pose.vy),
        rear_axle_acceleration_2d=StateVector2D(x=ego_pose.acceleration_x, y=ego_pose.acceleration_y),
    )


def extract_boxes(lidar_pc: LidarPc) -> List[Box3D]:
    """
    Extracts all boxes from a lidarpc.
    :param lidar_pc: Input lidarpc.
    :return: List of boxes contained in the lidarpc.
    """
    return [lidar_box.box() for lidar_box in lidar_pc.lidar_boxes]


def extract_tracked_objects(
    lidar_pc: LidarPc, future_trajectory_sampling: Optional[TrajectorySampling] = None
) -> TrackedObjects:
    """
    Extracts all boxes from a lidarpc.
    :param lidar_pc: Input lidarpc.
    :param future_trajectory_sampling: Sampling parameters for future predictions, if not provided, no future poses
    are extracted
    :return: Tracked objects contained in the lidarpc.
    """
    if future_trajectory_sampling:
        future_waypoints = get_interpolated_waypoints(lidar_pc, future_trajectory_sampling)
    else:
        future_waypoints = dict()

    return TrackedObjects(
        tracked_objects=[
            lidar_box.tracked_object(future_waypoints.get(lidar_box.track_token, None))
            for lidar_box in lidar_pc.lidar_boxes
        ]
    )


# Do not cache this method.
# This will keep old versions of the DB alive, which will lead to excessive memory allocations.
def lidarpc_next(lidarpc: LidarPc) -> Optional[LidarPc]:
    """
    Retrieve the next LidarPc from the database.
    :param lidarpc: Input lidarpc object.
    :return: Next lidarpc in the database.
    """
    next_lidarpc = lidarpc.next

    if next_lidarpc is None and lidarpc.next_token:
        log_lidarpcs = lidarpc.log.lidar_pcs
        next_lidarpc = log_lidarpcs[log_lidarpcs.index(lidarpc) + 1]

    return next_lidarpc


# Do not cache this method.
# This will keep old versions of the DB alive, which will lead to excessive memory allocations.
def lidarpc_prev(lidarpc: LidarPc) -> Optional[LidarPc]:
    """
    Retrieve the previous LidarPc from the database.
    :param lidarpc: Input lidarpc object.
    :return: Previous lidarpc in the database.
    """
    prev_lidarpc = lidarpc.prev

    if prev_lidarpc is None and lidarpc.prev_token:
        log_lidarpcs = lidarpc.log.lidar_pcs
        prev_lidarpc = log_lidarpcs[log_lidarpcs.index(lidarpc) - 1]

    return prev_lidarpc


# Do not cache this method.
# This will keep old versions of the DB alive, which will lead to excessive memory allocations.
def get_map_api(db: NuPlanDB, map_name: str) -> AbstractMap:
    """
    Retrieve the map API from a log.
    :param db: nuPlan database object.
    :param map_name: name of the map to load.
    :return: Retrieved map object.
    """
    return NuPlanMapFactory(db.maps_db).build_map_from_name(map_name)


def extract_lidarpc_tokens_as_scenario(
    db: NuPlanDB,
    anchor_timestamp: float,
    scenario_extraction_info: ScenarioExtractionInfo,
) -> List[str]:
    """
    Extract a list of lidarpc tokens that form a scenario around an anchor timestamp.
    :param db: Object providing DB access.
    :param anchor_timestamp: Timestamp of Lidarpc representing the start of the scenario.
    :param scenario_extraction_info: Structure containing information used to extract the scenario.
    :return: List of extracted lidarpc tokens representing the scenario.
    """
    start_timestamp = anchor_timestamp + scenario_extraction_info.extraction_offset * 1e6
    end_timestamp = start_timestamp + scenario_extraction_info.scenario_duration * 1e6

    lidarpcs = (
        db.session.query(LidarPc)
        .filter(LidarPc.timestamp > start_timestamp, LidarPc.timestamp < end_timestamp)
        .order_by(LidarPc.timestamp.asc())
        .all()
    )

    subsample_step = int(1.0 / scenario_extraction_info.subsample_ratio)
    if subsample_step < len(lidarpcs):
        lidarpcs = lidarpcs[::subsample_step]

    lidarpc_tokens = [lidarpc.token for lidarpc in lidarpcs]

    return lidarpc_tokens
