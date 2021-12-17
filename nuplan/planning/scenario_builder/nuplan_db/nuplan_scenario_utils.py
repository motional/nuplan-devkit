from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

from cachetools import LRUCache, cached
from cachetools.keys import hashkey

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.database.nuplan_db.models import Lidar, LidarPc, Log, Scene
from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

logger = logging.getLogger(__name__)

LIDAR_PC_CACHE = 8 * 2 ** 10  # 8K
DEFAULT_SCENARIO_DURATION: float = 20.0  # [s] duration of the scenario (e.g. extract 20s from when the event occurred)
DEFAULT_EXTRACTION_OFFSET: float = 0.0  # [s] offset of the scenario (e.g. start at -5s from when the event occurred)


@dataclass(frozen=True)
class ScenarioExtractionInfo:
    """
    Structure containing information used to extract a scenario (lidarpc sequence).
    """
    scenario_duration: float = DEFAULT_SCENARIO_DURATION  # [s] duration of the scenario
    extraction_offset: float = DEFAULT_EXTRACTION_OFFSET  # [s] offset of the scenario


class ScenarioMapping:
    """
    Structure that maps each scenario type to instructions used in extracting it.
    """

    def __init__(self, scenario_map: Dict[str, Tuple[float, float]]) -> None:
        self.mapping = {name: ScenarioExtractionInfo(value[0], value[1]) for name, value in scenario_map.items()}

    def get_extraction_info(self, scenario_type: str) -> Optional[ScenarioExtractionInfo]:
        """
        Accesses the scenario mapping using a query scenario type.
        If the scenario type is not found, a default extraction info object is returned.
        :param scenario_type: Scenario type to query for.
        :return: Scenario extraction information for the queried scenario type.
        """
        return self.mapping[scenario_type] if scenario_type in self.mapping else ScenarioExtractionInfo()


def extract_scenes_from_log(log: Log, first_valid_idx: int = 2, last_valid_idx: int = -2) -> List[Scene]:
    """
    Retrieves all valid scenes from a log.
    :param log: Log to get scenes from.
    :param first_valid_idx: Index of first valid scene.
    :param last_valid_idx: Index of last valid scene.
    :return: Retrieved scenes.
    """
    return cast(List[Scene], log.scenes[first_valid_idx:last_valid_idx])


def extract_lidar_pcs_from_scenes(scenes: List[Scene]) -> List[LidarPc]:
    """
    Retrieves all lidar pcs from a set of scenes.
    :param scenes: Scenes to get lidarpcs from.
    :return: Retrieved lidarpcs.
    """
    return [lidar_pc.token for scene in scenes for lidar_pc in scene.lidar_pcs]


def flatten_scenarios(scenarios: List[AbstractScenario]) -> List[AbstractScenario]:
    """
    Flatten scenarios (e.g. 10x 20-samples scenarios to 200x 1-sample scenarios).
    This is useful when training in open-loop where each data sample is one point in time.
    :param scenarios: List of scenarios to flatten.
    :return: List of flattened scenarios.
    """
    return [sample for scenario in scenarios for sample in scenario.flatten()]


def filter_invalid_goals(scenarios: List[AbstractScenario]) -> List[AbstractScenario]:
    """
    Filter scenarios that contains invalid mission goals.
    :param scenarios: List of scenarios to filter.
    :return: List of filtered scenarios.
    """
    return [scenario for scenario in scenarios if scenario.get_mission_goal()]


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
    :return: Instantiated state se2.
    """
    ego_pose = lidar_pc.ego_pose
    return StateSE2(ego_pose.x, ego_pose.y, ego_pose.quaternion.yaw_pitch_roll[0])


@cached(LRUCache(maxsize=LIDAR_PC_CACHE), key=lambda lidarpc: hashkey(lidarpc.token))  # type: ignore
def lidarpc_to_ego_state(lidar_pc: LidarPc) -> EgoState:
    """
    Converts a LidarPc to an EgoState object.
    :param lidar_pc: Input lidar pc.
    :return: Instantiated ego state.
    """
    ego_pose = lidar_pc.ego_pose
    return EgoState.from_raw_params(StateSE2(ego_pose.x, ego_pose.y, ego_pose.quaternion.yaw_pitch_roll[0]),
                                    tire_steering_angle=0.0,
                                    time_point=TimePoint(ego_pose.timestamp),
                                    velocity_2d=StateVector2D(x=ego_pose.vx, y=ego_pose.vy),
                                    acceleration_2d=StateVector2D(x=ego_pose.acceleration_x,
                                                                  y=ego_pose.acceleration_y))


def extract_boxes(lidar_pc: LidarPc) -> List[Box3D]:
    """
    Extracts all boxes from a lidarpc.
    :param lidar_pc: Input lidarpc.
    :return: List of boxes contained in the lidarpc.
    """
    return [lidar_box.box() for lidar_box in lidar_pc.lidar_boxes]


def extract_tracked_objects(lidar_pc: LidarPc) -> TrackedObjects:
    """
    Extracts all boxes from a lidarpc.
    :param lidar_pc: Input lidarpc.
    :return: Tracked objects contained in the lidarpc.
    """
    return TrackedObjects(agents=[lidar_box.agent() for lidar_box in lidar_pc.lidar_boxes])


@cached(LRUCache(maxsize=LIDAR_PC_CACHE),
        key=lambda lidarpc, db: hashkey(lidarpc.token, db.version, db.data_root))  # type: ignore
def lidarpc_next(lidarpc: LidarPc, db: NuPlanDB) -> Optional[LidarPc]:
    """
    Retrieves the next LidarPc from the database.
    :param lidarpc: input sample/lidarpc object.
    :return: next sample/lidarpc.
    """
    next_lidarpc = lidarpc.next

    if next_lidarpc is None and lidarpc.next_token:
        log_lidarpcs = lidarpc.log.lidar_pcs
        next_lidarpc = log_lidarpcs[log_lidarpcs.index(lidarpc) + 1]

    return next_lidarpc


@cached(LRUCache(maxsize=LIDAR_PC_CACHE),
        key=lambda lidarpc, db: hashkey(lidarpc.token, db.version, db.data_root))  # type: ignore
def lidarpc_prev(lidarpc: LidarPc, db: NuPlanDB) -> Optional[LidarPc]:
    """
    Retrieves the previous LidarPc from the database.
    :param lidarpc: input sample/lidarpc object.
    :return: previous sample/lidarpc.
    """
    prev_lidarpc = lidarpc.prev

    if prev_lidarpc is None and lidarpc.prev_token:
        log_lidarpcs = lidarpc.log.lidar_pcs
        prev_lidarpc = log_lidarpcs[log_lidarpcs.index(lidarpc) - 1]

    return prev_lidarpc


@cached(LRUCache(maxsize=8), key=lambda db, map_name: hashkey(map_name, db.version, db.data_root))  # type: ignore
def get_map_api(db: NuPlanDB, map_name: str) -> AbstractMap:
    """
    Retrieves the map API from a log.
    :param db: Databae object..
    :param map_name: name of the map to load.
    :return: Retrieved map object.
    """
    return NuPlanMapFactory(db.maps_db).build_map_from_name(map_name)


def extract_lidarpc_tokens_as_scenario(db: NuPlanDB, anchor_lidarpc: LidarPc,
                                       scenario_extraction_info: ScenarioExtractionInfo,
                                       subsample_ratio: Optional[float]) -> List[str]:
    """
    Extract a list of lidarpc tokens that form a scenario using an anchor lidarpc token.
    :param db: Object providing DB access.
    :param anchor_lidarpc: Lidarpc token representing the start of the scenario.
    :param scenario_extraction_info: Structure containing information used to extract the scenario.
    :param subsample_ratio: Ratio used sub-sample the scenario (e.g. from 20Hz to 2Hz using a 0.1 ratio).
    :return: List of extracted lidarpc tokens representing the scenario.
    """
    start_timestamp = anchor_lidarpc.timestamp + scenario_extraction_info.extraction_offset * 1e6
    end_timestamp = start_timestamp + scenario_extraction_info.scenario_duration * 1e6

    lidarpcs = db.session.query(LidarPc). \
        filter(
            Lidar.log_token == anchor_lidarpc.lidar.log_token,
            LidarPc.timestamp > start_timestamp,
            LidarPc.timestamp < end_timestamp). \
        order_by(LidarPc.timestamp.asc()) \
        .all()

    if subsample_ratio:
        subsample_step = int(1. / subsample_ratio)
        if subsample_step < len(lidarpcs):
            lidarpcs = lidarpcs[::subsample_step]

    lidarpc_tokens = [lidarpc.token for lidarpc in lidarpcs]

    return lidarpc_tokens
