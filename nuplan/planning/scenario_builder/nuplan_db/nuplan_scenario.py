from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType, Transform
from nuplan.database.nuplan_db.models import LidarPc
from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo, extract_boxes, \
    extract_lidarpc_tokens_as_scenario, extract_tracked_objects, get_map_api, get_time_stamp_from_lidar_pc, \
    lidarpc_next, lidarpc_prev, lidarpc_to_ego_state, lidarpc_to_state_se2
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import Detections, DetectionsTracks, Sensors


class NuPlanScenario(AbstractScenario):
    def __init__(self,
                 db: NuPlanDB,
                 initial_lidar_token: str,
                 subsample_ratio: Optional[float],
                 scenario_type: str,
                 scenario_extraction_info: Optional[ScenarioExtractionInfo],
                 ego_vehicle_parameters: VehicleParameters) -> None:
        """
        NuPlan scenario class.
        :param db: Object that provides database access.
        :param initial_lidar_token: Token of the scenario's initial lidarpc.
        :param subsample_ratio: Subsample ratio relative to the database frequency. None means no subsample.
        :param scenario_type: Type of scenario (e.g. ego overtaking).
        :param scenario_extraction_info: Structure containing information used to extract the scenario.
            None means the scenario has no length and it is comprised only by the initial lidarpc.
        :param ego_vehicle_parameters: Structure containing the vehicle parameters.
        """
        self._db = db
        self._initial_lidar_token = initial_lidar_token
        self._subsample_ratio = subsample_ratio
        self._scenario_type = scenario_type
        self._scenario_extraction_info = scenario_extraction_info
        self._ego_vehicle_parameters = ego_vehicle_parameters

    def __reduce__(self) -> Tuple[Type[NuPlanScenario], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._db, self._initial_lidar_token, self._subsample_ratio,
                                 self._scenario_type, self._scenario_extraction_info, self._ego_vehicle_parameters))

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """ Inherited, see superclass. """
        return self._ego_vehicle_parameters

    def _get_lidar_pc_at_iteration(self, iteration: int) -> LidarPc:
        """
        Return lidar pc from iteration
        :param iteration: within scenario
        :return: LidarPc
        """
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration: {iteration} is out of bounds!"
        return self._db.lidar_pc[self._lidarpc_tokens[iteration]]

    @cached_property
    def _initial_lidarpc(self) -> LidarPc:
        """
        :return: initial pointcloud
        """
        return self._db.lidar_pc[self._initial_lidar_token]

    @cached_property
    def _lidarpc_tokens(self) -> List[str]:
        """
        :return: list of lidarpc tokens in the scenario
        """
        if self._scenario_extraction_info is None:
            return [self._initial_lidar_token]

        lidarpc_tokens = extract_lidarpc_tokens_as_scenario(self._db, self._initial_lidarpc,
                                                            self._scenario_extraction_info, self._subsample_ratio)

        return cast(List[str], lidarpc_tokens)

    @cached_property
    def _final_lidarpc(self) -> LidarPc:
        """
        :return: last pointcloud
        """
        return self._get_lidar_pc_at_iteration(self.get_number_of_iterations() - 1)

    def _get_next_lidarpc(self, lidarpc: LidarPc, next_idx: int) -> LidarPc:
        """ Inherited, see superclass. """

        for _ in range(next_idx):
            assert lidarpc is not None, f"Error while retrieving lidarpc {next_idx} steps into the future"
            lidarpc = lidarpc_next(lidarpc, self._db)

        return lidarpc

    def _get_prev_lidarpc(self, lidarpc: LidarPc, prev_idx: int) -> LidarPc:
        """ Inherited, see superclass. """

        for _ in range(prev_idx):
            assert lidarpc is not None, f"Error while retrieving lidarpc {prev_idx} steps into the past"
            lidarpc = lidarpc_prev(lidarpc, self._db)

        return lidarpc

    @cached_property
    def _mission_goal(self) -> Optional[StateSE2]:
        """
        return: Mission goal based on the initial lidar pc.
        """
        pose_token = self._initial_lidarpc.scene.goal_ego_pose_token

        if pose_token is None:
            return None

        mission_pose = self._db.ego_pose[pose_token]
        return StateSE2(mission_pose.x, mission_pose.y, mission_pose.quaternion.yaw_pitch_roll[0])

    @property
    def token(self) -> str:
        """ Inherited, see superclass. """
        return self._initial_lidar_token

    @cached_property
    def log_name(self) -> str:
        """ Inherited, see superclass. """
        return cast(str, self._initial_lidarpc.log.logfile)

    @property
    def scenario_name(self) -> str:
        """ Inherited, see superclass. """
        return self.token

    @property
    def scenario_type(self) -> str:
        """ Inherited, see superclass. """
        return self._scenario_type

    @property
    def map_api(self) -> AbstractMap:
        """ Inherited, see superclass. """
        return get_map_api(self._db, self._initial_lidarpc.log.map_version)

    @property
    def database_interval(self) -> float:
        """ Inherited, see superclass. """
        return 0.05  # 20Hz

    def flatten(self) -> List[NuPlanScenario]:
        """ Inherited, see superclass. """
        kwargs: Dict[str, Any] = {
            'db': self._db,
            'subsample_ratio': self._subsample_ratio,
            'scenario_type': self._scenario_type,
            'scenario_extraction_info': None,  # a flattened scenario cannot be further extracted
            'ego_vehicle_parameters': self._ego_vehicle_parameters,
        }

        return [self.__class__(initial_lidar_token=token, **kwargs) for token in self._lidarpc_tokens]

    def get_number_of_iterations(self) -> int:
        """ Inherited, see superclass. """
        return len(self._lidarpc_tokens)

    def get_lidar_to_ego_transform(self) -> Transform:
        """ Inherited, see superclass. """
        return self._initial_lidarpc.lidar.trans_matrix

    def get_mission_goal(self) -> Optional[StateSE2]:
        """ Inherited, see superclass. """
        return self._mission_goal

    def get_expert_goal_state(self) -> StateSE2:
        """ Inherited, see superclass. """
        return lidarpc_to_state_se2(self._final_lidarpc)

    def get_time_point(self, iteration: int) -> TimePoint:
        """ Inherited, see superclass. """
        return TimePoint(get_time_stamp_from_lidar_pc(self._get_lidar_pc_at_iteration(iteration)))

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """ Inherited, see superclass. """
        return lidarpc_to_ego_state(self._get_lidar_pc_at_iteration(iteration))

    def get_detections_at_iteration(self, iteration: int) -> Detections:
        """ Inherited, see superclass. """
        return Detections(boxes=extract_boxes(self._get_lidar_pc_at_iteration(iteration)))

    def get_tracked_objects_at_iteration(self, iteration: int) -> DetectionsTracks:
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"
        return DetectionsTracks(extract_tracked_objects(self._get_lidar_pc_at_iteration(iteration)))

    def get_sensors_at_iteration(self, iteration: int) -> Sensors:
        """ Inherited, see superclass. """
        lidar_pc = self._get_lidar_pc_at_iteration(iteration)
        return Sensors(pointcloud=lidar_pc.load().points.T)

    def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) \
            -> List[TimePoint]:
        """ Inherited, see superclass. """
        return [TimePoint(get_time_stamp_from_lidar_pc(lidar_pc)) for lidar_pc in
                self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True)]

    def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) \
            -> List[TimePoint]:
        """ Inherited, see superclass. """
        return [TimePoint(get_time_stamp_from_lidar_pc(lidar_pc)) for lidar_pc in
                self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False)]

    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) \
            -> List[EgoState]:
        """ Inherited, see superclass. """
        return [lidarpc_to_ego_state(lidar_pc) for lidar_pc in
                self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False)]

    def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) \
            -> List[EgoState]:
        """ Inherited, see superclass. """
        return [lidarpc_to_ego_state(lidar_pc) for lidar_pc in
                self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True)]

    def get_past_detections(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) \
            -> List[Detections]:
        """ Inherited, see superclass. """
        return [Detections(boxes=extract_boxes(lidar_pc)) for lidar_pc in
                self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False)]

    def get_future_detections(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) \
            -> List[Detections]:
        """ Inherited, see superclass. """
        return [Detections(boxes=extract_boxes(lidar_pc)) for lidar_pc in
                self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True)]

    def get_past_tracked_objects(self, iteration: int, num_samples: int, time_horizon: float) -> List[DetectionsTracks]:
        """ Inherited, see superclass. """
        return [DetectionsTracks(extract_tracked_objects(lidar_pc)) for lidar_pc in
                self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False)]

    def get_future_tracked_objects(self, iteration: int, num_samples: int, time_horizon: float) \
            -> List[DetectionsTracks]:
        return [DetectionsTracks(extract_tracked_objects(lidar_pc)) for lidar_pc in
                self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True)]

    def get_past_sensors(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) \
            -> List[Sensors]:
        """ Inherited, see superclass. """
        return [Sensors(pointcloud=lidar_pc.load().points.T) for lidar_pc in
                self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False)]

    def get_traffic_light_status_at_iteration(self, iteration: int) -> List[TrafficLightStatusData]:
        """ Inherited, see superclass. """

        lidar_pc_token = self._lidarpc_tokens[iteration]
        traffic_light_status = self._db.traffic_light_status.select_many(lidar_pc_token=lidar_pc_token)
        traffic_light_status_data = [TrafficLightStatusData(
            status=TrafficLightStatusType[traffic_light.status.upper()],
            stop_line_id=traffic_light.stop_line_id,
            lane_connector_id=traffic_light.lane_connector_id,
            timestamp=self._get_lidar_pc_at_iteration(iteration).timestamp) for traffic_light in traffic_light_status]
        return traffic_light_status_data

    def _find_matching_lidar_pcs(self, iteration: int, num_samples: Optional[int], time_horizon: float,
                                 look_into_future: bool) -> List[LidarPc]:
        """
        Find the best matching lidar_pcs to the desired samples and time horizon
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future, if None it will be deduced from the DB
        :param time_horizon: the desired horizon to the future
        :param look_into_future: if True, we will iterate into next lidar_pc otherwise we will iterate through prev
        :return: lidar_pcs matching to database indices
        """
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self.database_interval)
        return [self._get_next_lidarpc(self._get_lidar_pc_at_iteration(iteration), idx)
                for idx in indices] if look_into_future else \
            [self._get_prev_lidarpc(self._get_lidar_pc_at_iteration(iteration), idx) for idx in reversed(indices)]
