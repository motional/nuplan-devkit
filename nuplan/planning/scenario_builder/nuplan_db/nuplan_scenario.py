from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Any, Generator, List, Optional, Set, Tuple, Type, cast

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatuses, Transform
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.nuplan_map.utils import get_roadblock_ids_from_trajectory
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_ego_state_for_lidarpc_token_from_db,
    get_end_sensor_time_from_db,
    get_images_from_lidar_tokens,
    get_mission_goal_for_sensor_data_token_from_db,
    get_roadblock_ids_for_lidarpc_token_from_db,
    get_sampled_ego_states_from_db,
    get_sampled_lidarpcs_from_db,
    get_sensor_data_from_sensor_data_tokens_from_db,
    get_sensor_data_token_timestamp_from_db,
    get_sensor_transform_matrix_for_sensor_data_token_from_db,
    get_statese2_for_lidarpc_token_from_db,
    get_traffic_light_status_for_lidarpc_token_from_db,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioExtractionInfo,
    absolute_path_to_log_name,
    download_file_if_necessary,
    extract_sensor_tokens_as_scenario,
    extract_tracked_objects,
    extract_tracked_objects_within_time_window,
    load_image,
    load_point_cloud,
)
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import (
    CameraChannel,
    DetectionsTracks,
    LidarChannel,
    SensorChannel,
    Sensors,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class NuPlanScenario(AbstractScenario):
    """Scenario implementation for the nuPlan dataset that is used in training and simulation."""

    def __init__(
        self,
        data_root: str,
        log_file_load_path: str,
        initial_lidar_token: str,
        initial_lidar_timestamp: int,
        scenario_type: str,
        map_root: str,
        map_version: str,
        map_name: str,
        scenario_extraction_info: Optional[ScenarioExtractionInfo],
        ego_vehicle_parameters: VehicleParameters,
        sensor_root: Optional[str] = None,
    ) -> None:
        """
        Initialize the nuPlan scenario.
        :param data_root: The prefix for the log file. e.g. "/data/root/nuplan". For remote paths, this is where the file will be downloaded if necessary.
        :param log_file_load_path: Name of the log that this scenario belongs to. e.g. "/data/sets/nuplan-v1.1/splits/mini/2021.07.16.20.45.29_veh-35_01095_01486.db", "s3://path/to/db.db"
        :param initial_lidar_token: Token of the scenario's initial lidarpc.
        :param initial_lidar_timestamp: The timestamp of the initial lidarpc.
        :param scenario_type: Type of scenario (e.g. ego overtaking).
        :param map_root: The root path for the map db
        :param map_version: The version of maps to load
        :param map_name: The map name to use for the scenario
        :param scenario_extraction_info: Structure containing information used to extract the scenario.
            None means the scenario has no length and it is comprised only by the initial lidarpc.
        :param ego_vehicle_parameters: Structure containing the vehicle parameters.
        :param sensor_root: The root path for the sensor blobs.
        """
        # Lazily Create
        self._local_store: Optional[LocalStore] = None
        self._remote_store: Optional[S3Store] = None

        self._data_root = data_root
        self._log_file_load_path = log_file_load_path
        self._initial_lidar_token = initial_lidar_token
        self._initial_lidar_timestamp = initial_lidar_timestamp
        self._scenario_type = scenario_type
        self._map_root = map_root
        self._map_version = map_version
        self._map_name = map_name
        self._scenario_extraction_info = scenario_extraction_info
        self._ego_vehicle_parameters = ego_vehicle_parameters
        self._sensor_root = sensor_root

        # If scenario extraction info is provided, check that the subsample ratio is valid
        if self._scenario_extraction_info is not None:
            skip_rows = 1.0 / self._scenario_extraction_info.subsample_ratio
            if abs(int(skip_rows) - skip_rows) > 1e-3:
                raise ValueError(
                    f"Subsample ratio is not valid. Must resolve to an integer number of skipping rows, instead received {self._scenario_extraction_info.subsample_ratio}, which would skip {skip_rows} rows."
                )

        # The interval between successive rows in the DB.
        # This is necessary for functions that sample the rows, such as get_ego_future_trajectory
        self._database_row_interval = 0.05

        # Typically, the log file will already be downloaded by the scenario_builder by this point
        #   So most of the time, this should be a trivial translation.
        #
        # However, in the situation in which a scenario is serialized, then deserialized on another machine,
        #   The log file may not be downloaded.
        #
        # So, we must check and download the file here as well.
        self._log_file = download_file_if_necessary(self._data_root, self._log_file_load_path)
        self._log_name: str = absolute_path_to_log_name(self._log_file)

    def __reduce__(self) -> Tuple[Type[NuPlanScenario], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (
            self.__class__,
            (
                self._data_root,
                self._log_file_load_path,
                self._initial_lidar_token,
                self._initial_lidar_timestamp,
                self._scenario_type,
                self._map_root,
                self._map_version,
                self._map_name,
                self._scenario_extraction_info,
                self._ego_vehicle_parameters,
                self._sensor_root,
            ),
        )

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return self._ego_vehicle_parameters

    @cached_property
    def _lidarpc_tokens(self) -> List[str]:
        """
        :return: list of lidarpc tokens in the scenario
        """
        if self._scenario_extraction_info is None:
            return [self._initial_lidar_token]

        lidarpc_tokens = list(
            extract_sensor_tokens_as_scenario(
                self._log_file,
                get_lidarpc_sensor_data(),
                self._initial_lidar_timestamp,
                self._scenario_extraction_info,
            )
        )

        return cast(List[str], lidarpc_tokens)

    @cached_property
    def _route_roadblock_ids(self) -> List[str]:
        """
        return: Route roadblock ids extracted from expert trajectory.
        """
        expert_trajectory = list(self._extract_expert_trajectory())
        return get_roadblock_ids_from_trajectory(self.map_api, expert_trajectory)  # type: ignore

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._initial_lidar_token

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        # e.g. "2021.07.16.20.45.29_veh-35_01095_01486.db"
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self.token

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_type

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        return get_maps_api(self._map_root, self._map_version, self._map_name)

    @property
    def map_root(self) -> str:
        """Get the map root folder."""
        return self._map_root

    @property
    def map_version(self) -> str:
        """Get the map version."""
        return self._map_version

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        if self._scenario_extraction_info is None:
            return 0.05  # 20Hz
        return float(0.05 / self._scenario_extraction_info.subsample_ratio)

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return len(self._lidarpc_tokens)

    def get_lidar_to_ego_transform(self) -> Transform:
        """Inherited, see superclass."""
        return get_sensor_transform_matrix_for_sensor_data_token_from_db(
            self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token
        )

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Inherited, see superclass."""
        return get_mission_goal_for_sensor_data_token_from_db(
            self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token
        )

    def get_route_roadblock_ids(self) -> List[str]:
        """Inherited, see superclass."""
        roadblock_ids = get_roadblock_ids_for_lidarpc_token_from_db(self._log_file, self._initial_lidar_token)
        assert roadblock_ids is not None, "Unable to find Roadblock ids for current scenario"
        return cast(List[str], roadblock_ids)

    def get_expert_goal_state(self) -> StateSE2:
        """Inherited, see superclass."""
        return get_statese2_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[-1])

    def get_time_point(self, iteration: int) -> TimePoint:
        """Inherited, see superclass."""
        return TimePoint(
            time_us=get_sensor_data_token_timestamp_from_db(
                self._log_file, get_lidarpc_sensor_data(), self._lidarpc_tokens[iteration]
            )
        )

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""
        return get_ego_state_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[iteration])

    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"
        return DetectionsTracks(
            extract_tracked_objects(self._lidarpc_tokens[iteration], self._log_file, future_trajectory_sampling)
        )

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"
        return DetectionsTracks(
            extract_tracked_objects_within_time_window(
                self._lidarpc_tokens[iteration],
                self._log_file,
                past_time_horizon,
                future_time_horizon,
                filter_track_tokens,
                future_trajectory_sampling,
            )
        )

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]] = None) -> Sensors:
        """Inherited, see superclass."""
        # To maintain backwards compatibility. We return lidar_pc by default.
        channels = [LidarChannel.MERGED_PC] if channels is None else channels

        lidar_pc = next(
            get_sensor_data_from_sensor_data_tokens_from_db(
                self._log_file, get_lidarpc_sensor_data(), LidarPc, [self._lidarpc_tokens[iteration]]
            )
        )
        return self._get_sensor_data_from_lidar_pc(cast(LidarPc, lidar_pc), channels)

    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TimePoint(lidar_pc.timestamp)

    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TimePoint(lidar_pc.timestamp)

    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)

        return cast(
            Generator[EgoState, None, None],
            get_sampled_ego_states_from_db(
                self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=False
            ),
        )

    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)

        return cast(
            Generator[EgoState, None, None],
            get_sampled_ego_states_from_db(
                self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=True
            ),
        )

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        # TODO: This can be made even more efficient with a batch query
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        # TODO: This can be made even more efficient with a batch query
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        """Inherited, see superclass."""
        # To maintain backwards compatibility. We return lidar_pc by default.
        channels = [LidarChannel.MERGED_PC] if channels is None else channels

        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield self._get_sensor_data_from_lidar_pc(lidar_pc, channels)

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Inherited, see superclass."""
        token = self._lidarpc_tokens[iteration]

        return cast(
            Generator[TrafficLightStatusData, None, None],
            get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, token),
        )

    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets past traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the past.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the past.
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TrafficLightStatuses(
                list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token))
            )

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets future traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the future.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the future.
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TrafficLightStatuses(
                list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token))
            )

    def get_scenario_tokens(self) -> List[str]:
        """Return the list of lidarpc tokens from the DB that are contained in the scenario."""
        return self._lidarpc_tokens

    def _find_matching_lidar_pcs(
        self, iteration: int, num_samples: Optional[int], time_horizon: float, look_into_future: bool
    ) -> Generator[LidarPc, None, None]:
        """
        Find the best matching lidar_pcs to the desired samples and time horizon
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future, if None it will be deduced from the DB
        :param time_horizon: the desired horizon to the future
        :param look_into_future: if True, we will iterate into next lidar_pc otherwise we will iterate through prev
        :return: lidar_pcs matching to database indices
        """
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)

        return cast(
            Generator[LidarPc, None, None],
            get_sampled_lidarpcs_from_db(
                self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, look_into_future
            ),
        )

    def _extract_expert_trajectory(self, max_future_seconds: int = 60) -> Generator[EgoState, None, None]:
        """
        Extract expert trajectory with specified time parameters. If initial lidar pc does not have enough history/future
            only available time will be extracted
        :param max_future_seconds: time to future which should be considered for route extraction [s]
        :return: list of expert ego states
        """
        minimal_required_future_time_available = 0.5

        # Extract Future
        end_log_time_us = get_end_sensor_time_from_db(self._log_file, get_lidarpc_sensor_data())
        max_future_time = min((end_log_time_us - self._initial_lidar_timestamp) * 1e-6, max_future_seconds)

        if max_future_time < minimal_required_future_time_available:
            return

        for traj in self.get_ego_future_trajectory(0, max_future_time):
            yield traj

    def _create_blob_store_if_needed(self) -> Tuple[LocalStore, Optional[S3Store]]:
        """
        A convenience method that creates the blob stores if it's not already created.
        :return: The created or cached LocalStore and S3Store objects.
        """
        if self._local_store is not None and self._remote_store is not None:
            return self._local_store, self._remote_store

        if self._sensor_root is None:
            raise ValueError("sensor_root is not set. Please set the sensor_root to access sensor data.")
        Path(self._sensor_root).mkdir(exist_ok=True)
        self._local_store = LocalStore(self._sensor_root)
        if os.getenv("NUPLAN_DATA_STORE", "") == "s3":
            s3_url = os.getenv("NUPLAN_DATA_ROOT_S3_URL", "")
            self._remote_store = S3Store(os.path.join(s3_url, "sensor_blobs"), show_progress=True)

        return self._local_store, self._remote_store

    def _get_sensor_data_from_lidar_pc(self, lidar_pc: LidarPc, channels: List[SensorChannel]) -> Sensors:
        """
        Loads Sensor data given a database LidarPC object.
        :param lidar_pc: The lidar_pc for which to grab the point cloud.
        :param channels: The sensor channels to return.
        :return: The corresponding sensor data.
        """
        local_store, remote_store = self._create_blob_store_if_needed()

        retrieved_images = get_images_from_lidar_tokens(
            self._log_file, [lidar_pc.token], [cast(str, channel.value) for channel in channels]
        )
        lidar_pcs = (
            {LidarChannel.MERGED_PC: load_point_cloud(cast(LidarPc, lidar_pc), local_store, remote_store)}
            if LidarChannel.MERGED_PC in channels
            else None
        )

        images = {
            CameraChannel[image.channel]: load_image(image, local_store, remote_store) for image in retrieved_images
        }

        return Sensors(pointcloud=lidar_pcs, images=images if images else None)
