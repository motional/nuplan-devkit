from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Generator, List, Optional, Set, Tuple, Union, cast

import nuplan.database.nuplan_db.image as ImageDBRow
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.geometry.interpolate_state import interpolate_future_waypoints
from nuplan.database.common.blob_store.creator import BlobStoreCreator
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplan_db_utils import SensorDataSource, get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_future_waypoints_for_agents_from_db,
    get_sampled_sensor_tokens_in_time_window_from_db,
    get_sensor_data_token_timestamp_from_db,
    get_tracked_objects_for_lidarpc_token_from_db,
    get_tracked_objects_within_time_interval_from_db,
)
from nuplan.database.utils.image import Image
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)

LIDAR_PC_CACHE = 16 * 2**10  # 16K

DEFAULT_SCENARIO_NAME = 'unknown'  # name of scenario (e.g. ego overtaking)
DEFAULT_SCENARIO_DURATION = 20.0  # [s] duration of the scenario (e.g. extract 20s from when the event occurred)
DEFAULT_EXTRACTION_OFFSET = 0.0  # [s] offset of the scenario (e.g. start at -5s from when the event occurred)
DEFAULT_SUBSAMPLE_RATIO = 1.0  # ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)

NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', "/data/sets/nuplan/")


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

    def __init__(
        self,
        scenario_map: Dict[str, Union[Tuple[float, float, float], Tuple[float, float]]],
        subsample_ratio_override: Optional[float],
    ) -> None:
        """
        Initializes the scenario mapping class.
        :param scenario_map: Dictionary with scenario name/type as keys and
                             tuples of (scenario duration, extraction offset, subsample ratio) as values.
        :subsample_ratio_override: The override for the subsample ratio if not provided.
        """
        self.mapping: Dict[str, ScenarioExtractionInfo] = {}
        self.subsample_ratio_override = (
            subsample_ratio_override if subsample_ratio_override is not None else DEFAULT_SUBSAMPLE_RATIO
        )

        for name in scenario_map:
            this_ratio: float = scenario_map[name][2] if len(scenario_map[name]) == 3 else self.subsample_ratio_override  # type: ignore

            self.mapping[name] = ScenarioExtractionInfo(
                scenario_name=name,
                scenario_duration=scenario_map[name][0],
                extraction_offset=scenario_map[name][1],
                subsample_ratio=this_ratio,
            )

    def get_extraction_info(self, scenario_type: str) -> Optional[ScenarioExtractionInfo]:
        """
        Accesses the scenario mapping using a query scenario type.
        If the scenario type is not found, a default extraction info object is returned.
        :param scenario_type: Scenario type to query for.
        :return: Scenario extraction information for the queried scenario type.
        """
        return (
            self.mapping[scenario_type]
            if scenario_type in self.mapping
            else ScenarioExtractionInfo(subsample_ratio=self.subsample_ratio_override)
        )


def download_file_if_necessary(data_root: str, potentially_remote_path: str, verbose: bool = False) -> str:
    """
    Downloads the db file if necessary.
    :param data_root: Path's data root.
    :param potentially_remote_path: The path from which to download the file.
    :param verbose: Verbosity level.
    :return: The local path for the file.
    """
    # If the file path is a local directory and exists, then return that.
    # e.g. /data/sets/nuplan/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    if os.path.exists(potentially_remote_path):
        return potentially_remote_path

    log_name = absolute_path_to_log_name(potentially_remote_path)
    download_name = log_name + ".db"

    # TODO: CacheStore seems to be buggy.
    # Behavior seems to be different on our cluster vs locally regarding downloaded file paths.
    #
    # Use the underlying stores manually.
    os.makedirs(data_root, exist_ok=True)
    local_store = LocalStore(data_root)

    if not local_store.exists(download_name):
        blob_store = BlobStoreCreator.create_nuplandb(data_root, verbose=verbose)

        # If we have no matches, download the file.
        logger.info("DB path not found. Downloading to %s..." % download_name)
        start_time = time.time()

        # Infer remote path from local path
        remote_key = potentially_remote_path
        if not remote_key.startswith("s3://"):
            fixed_local_path = convert_legacy_nuplan_path_to_latest(potentially_remote_path)
            remote_key = infer_remote_key_from_local_path(fixed_local_path)

        content = blob_store.get(remote_key)
        local_store.put(download_name, content)
        logger.info("Downloading db file took %.2f seconds." % (time.time() - start_time))

    return os.path.join(data_root, download_name)


def convert_legacy_nuplan_path_to_latest(legacy_path: str, nuplan_data_root: Optional[str] = None) -> str:
    """
    Converts known legacy nuPlan path formats to the latest version.
    Examples:
    - data_root: /data/sets/nuplan/
      in:  /data/sets/nuplan/nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
      out: /data/sets/nuplan/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    :param legacy_path: Legacy path to convert.
    :param nuplan_data_root: Optional custom nuPlan data root directory. When None is supplied, the NUPLAN_DATA_ROOT environment variable will be used.
    :return: Converted input path.
    """
    # sanity check, exit early if no version is found
    if legacy_path.find("nuplan-v") == -1:
        raise ValueError("nuPlan DB path should contain db version in it (e.g: nuplan-v1.1)")

    # remove data root, we don't care about them when comparing
    # intended effect:
    # /data/sets/nuplan/nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db -> /nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    if nuplan_data_root is None:
        nuplan_data_root = NUPLAN_DATA_ROOT
    prefix_removed = legacy_path.removeprefix(nuplan_data_root)

    # make sure path doesn't start with '/' due to possible differences in NUPLAN_DATA_ROOT's definition
    # i.e: NUPLAN_DATA_ROOT may or may not end with `/'
    # intended effect:
    # - /nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db -> nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    # - nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db -> nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    prefix_removed = prefix_removed.lstrip("/")

    # our candidate return value
    prefix_removed_path = Path(prefix_removed)

    # insert splits if `splits` directory is not found
    # intended effect:
    # - nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db -> nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    if prefix_removed.find("splits") == -1:
        path_parts = list(prefix_removed_path.parts)
        version_directory_index = min(
            idx for idx, directory_name in enumerate(path_parts) if "nuplan-v" in directory_name
        )
        path_parts.insert(version_directory_index + 1, "splits")
        prefix_removed_path = Path("/".join(path_parts))

    return_path = Path(nuplan_data_root) / prefix_removed_path

    return str(return_path)


def infer_remote_key_from_local_path(local_path: str, nuplan_data_root: Optional[str] = None) -> str:
    """
    Try to infer a file's remote key on s3 based on its local path.
    Examples:
    - nuplan_data_root: /data/sets/nuplan/
      in:  /data/sets/nuplan/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
      out: splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    :param local_path: Local path of the file.
    :param nuplan_data_root: Optional custom nuPlan data root directory. When None is supplied, the NUPLAN_DATA_ROOT environment variable will be used.
    :return: Inferred remote key.
    """
    # remove local data root
    # intended effect:
    # /data/sets/nuplan/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db -> /nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    if nuplan_data_root is None:
        nuplan_data_root = NUPLAN_DATA_ROOT
    remote_key = local_path.removeprefix(nuplan_data_root)

    # make sure path doesn't start with '/' due to possible differences in NUPLAN_DATA_ROOT's definition
    # i.e: NUPLAN_DATA_ROOT may or may not end with `/'
    # intended effect:
    # - /nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db -> nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    # - nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db -> nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    remote_key = remote_key.lstrip("/")

    # Remove nuPlan version from path
    # intended effect:
    # - nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db -> splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    if remote_key.startswith("nuplan-v"):
        remote_key_as_path = Path(remote_key)
        remote_key_as_path = Path(*remote_key_as_path.parts[1:])
        remote_key = str(remote_key_as_path)

    # final result: splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db
    return remote_key


def _process_future_trajectories_for_windowed_agents(
    log_file: str,
    tracked_objects: List[TrackedObject],
    agent_indexes: Dict[int, Dict[str, int]],
    future_trajectory_sampling: TrajectorySampling,
) -> List[TrackedObject]:
    """
    A helper method to interpolate and parse the future trajectories for windowed agents.
    :param log_file: The log file to query.
    :param tracked_objects: The tracked objects to parse.
    :param agent_indexes: A mapping of [timestamp, [track_token, tracked_object_idx]]
    :param future_trajectory_sampling: The future trajectory sampling to use for future waypoints.
    :return: The tracked objects with predicted trajectories included.
    """
    agent_future_trajectories: Dict[int, Dict[str, List[Waypoint]]] = {}
    for timestamp in agent_indexes:
        agent_future_trajectories[timestamp] = {}

        for token in agent_indexes[timestamp]:
            agent_future_trajectories[timestamp][token] = []

    for timestamp_time in agent_future_trajectories:
        end_time = timestamp_time + int(
            1e6 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length)
        )

        # TODO: This is somewhat inefficient because the resampling should happen in SQL layer

        for track_token, waypoint in get_future_waypoints_for_agents_from_db(
            log_file, list(agent_indexes[timestamp_time].keys()), timestamp_time, end_time
        ):
            agent_future_trajectories[timestamp_time][track_token].append(waypoint)

    for timestamp in agent_future_trajectories:
        for key in agent_future_trajectories[timestamp]:
            # We can only interpolate waypoints if there is more than one in the future.
            if len(agent_future_trajectories[timestamp][key]) == 1:
                tracked_objects[agent_indexes[timestamp][key]]._predictions = [
                    PredictedTrajectory(1.0, agent_future_trajectories[timestamp][key])
                ]
            elif len(agent_future_trajectories[timestamp][key]) > 1:
                tracked_objects[agent_indexes[timestamp][key]]._predictions = [
                    PredictedTrajectory(
                        1.0,
                        interpolate_future_waypoints(
                            agent_future_trajectories[timestamp][key],
                            future_trajectory_sampling.time_horizon,
                            future_trajectory_sampling.interval_length,
                        ),
                    )
                ]

    return tracked_objects


def extract_tracked_objects_within_time_window(
    token: str,
    log_file: str,
    past_time_horizon: float,
    future_time_horizon: float,
    filter_track_tokens: Optional[Set[str]] = None,
    future_trajectory_sampling: Optional[TrajectorySampling] = None,
) -> TrackedObjects:
    """
    Extracts the tracked objects in a time window centered on a token.
    :param token: The token on which to center the time window.
    :param past_time_horizon: The time in the past for which to search.
    :param future_time_horizon: The time in the future for which to search.
    :param filter_track_tokens: If provided, objects with track_tokens missing from the set will be excluded.
    :param future_trajectory_sampling: If provided, the future trajectory sampling to use for future waypoints.
    :return: The retrieved TrackedObjects.
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[int, Dict[str, int]] = {}

    token_timestamp = get_sensor_data_token_timestamp_from_db(log_file, get_lidarpc_sensor_data(), token)
    start_time = int(token_timestamp - (1e6 * past_time_horizon))
    end_time = int(token_timestamp + (1e6 * future_time_horizon))

    for idx, tracked_object in enumerate(
        get_tracked_objects_within_time_interval_from_db(log_file, start_time, end_time, filter_track_tokens)
    ):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            if tracked_object.metadata.timestamp_us not in agent_indexes:
                agent_indexes[tracked_object.metadata.timestamp_us] = {}

            agent_indexes[tracked_object.metadata.timestamp_us][tracked_object.metadata.track_token] = idx
        tracked_objects.append(tracked_object)

    if future_trajectory_sampling:
        _process_future_trajectories_for_windowed_agents(
            log_file, tracked_objects, agent_indexes, future_trajectory_sampling
        )

    return TrackedObjects(tracked_objects=tracked_objects)


def extract_tracked_objects(
    token: str,
    log_file: str,
    future_trajectory_sampling: Optional[TrajectorySampling] = None,
) -> TrackedObjects:
    """
    Extracts all boxes from a lidarpc.
    :param lidar_pc: Input lidarpc.
    :param future_trajectory_sampling: If provided, the future trajectory sampling to use for future waypoints.
    :return: Tracked objects contained in the lidarpc.
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[str, int] = {}
    agent_future_trajectories: Dict[str, List[Waypoint]] = {}

    for idx, tracked_object in enumerate(get_tracked_objects_for_lidarpc_token_from_db(log_file, token)):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            agent_indexes[tracked_object.metadata.track_token] = idx
            agent_future_trajectories[tracked_object.metadata.track_token] = []
        tracked_objects.append(tracked_object)

    if future_trajectory_sampling and len(tracked_objects) > 0:
        timestamp_time = get_sensor_data_token_timestamp_from_db(log_file, get_lidarpc_sensor_data(), token)
        end_time = timestamp_time + int(
            1e6 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length)
        )

        # TODO: This is somewhat inefficient because the resampling should happen in SQL layer
        for track_token, waypoint in get_future_waypoints_for_agents_from_db(
            log_file, list(agent_indexes.keys()), timestamp_time, end_time
        ):
            agent_future_trajectories[track_token].append(waypoint)

        for key in agent_future_trajectories:
            # We can only interpolate waypoints if there is more than one in the future.
            if len(agent_future_trajectories[key]) == 1:
                tracked_objects[agent_indexes[key]]._predictions = [
                    PredictedTrajectory(1.0, agent_future_trajectories[key])
                ]
            elif len(agent_future_trajectories[key]) > 1:
                tracked_objects[agent_indexes[key]]._predictions = [
                    PredictedTrajectory(
                        1.0,
                        interpolate_future_waypoints(
                            agent_future_trajectories[key],
                            future_trajectory_sampling.time_horizon,
                            future_trajectory_sampling.interval_length,
                        ),
                    )
                ]

    return TrackedObjects(tracked_objects=tracked_objects)


def extract_sensor_tokens_as_scenario(
    log_file: str,
    sensor_data_source: SensorDataSource,
    anchor_timestamp: float,
    scenario_extraction_info: ScenarioExtractionInfo,
) -> Generator[str, None, None]:
    """
    Extract a list of sensor tokens that form a scenario around an anchor timestamp.
    :param log_file: The log file to access
    :param sensor_data_source: Parameters for querying the correct table.
    :param anchor_timestamp: Timestamp of Sensor representing the start of the scenario.
    :param scenario_extraction_info: Structure containing information used to extract the scenario.
    :return: List of extracted sensor tokens representing the scenario.
    """
    start_timestamp = int(anchor_timestamp + scenario_extraction_info.extraction_offset * 1e6)
    end_timestamp = int(start_timestamp + scenario_extraction_info.scenario_duration * 1e6)
    subsample_step = int(1.0 / scenario_extraction_info.subsample_ratio)

    return cast(
        Generator[str, None, None],
        get_sampled_sensor_tokens_in_time_window_from_db(
            log_file, sensor_data_source, start_timestamp, end_timestamp, subsample_step
        ),
    )


def absolute_path_to_log_name(absolute_path: str) -> str:
    """
    Gets the log name from the absolute path to a log file.
    E.g.
        input: data/sets/nuplan/nuplan-v1.1/splits/mini/2021.10.11.02.57.41_veh-50_01522_02088.db
        output: 2021.10.11.02.57.41_veh-50_01522_02088

        input: /tmp/abcdef
        output: abcdef
    :param absolute_path: The absolute path to a log file.
    :return: The log name.
    """
    filename = os.path.basename(absolute_path)

    # Files generated during caching do not end with ".db"
    # They have no extension.
    if filename.endswith(".db"):
        filename = os.path.splitext(filename)[0]
    return filename


def download_and_cache(key: str, local_store: LocalStore, remote_store: S3Store) -> Optional[BinaryIO]:
    """
    Downloads and cache the key given. This function assumes that the local and remotes stores are already configured.
    Data will be downloaded from the remote store's s3 bucket and saved relative to the data root of the local store.
    This method will initialize the scenario's blob store if it does not already exist.
    :param key: The key for which to grab the sensor data.
    :param local_store: Local blob store for loading blobs from local file system.
    :param remote_store: S3 blob store for loading blobs from AWS S3.
    :return: The sensor data.
    """
    if local_store.exists(key):
        return cast(BinaryIO, local_store.get(key))

    if remote_store is None:
        raise RuntimeError(
            "Remote store is not set and key was not found locally. Try setting NUPLAN_DATA_STORE to 's3'."
        )

    # Download and store data locally
    try:
        blob = remote_store.get(key)
        local_store.put(key, blob)
        return cast(BinaryIO, local_store.get(key))
    except RuntimeError as error:
        logging.warning(f"Could not find sensor data locally or remotely. Returning None\nCause: {error}")
        return None


def load_point_cloud(lidar_pc: LidarPc, local_store: LocalStore, remote_store: S3Store) -> Optional[LidarPointCloud]:
    """
    Loads a point cloud given a database LidarPC object.
    :param lidar_pc: The lidar_pc for which to grab the point cloud.
    :param local_store: Local blob store for loading blobs from local file system.
    :param remote_store: S3 blob store for loading blobs from AWS S3.
    :return: The corresponding point cloud.
    """
    file_type = lidar_pc.filename.split('.')[-1]
    blob = download_and_cache(lidar_pc.filename, local_store, remote_store)
    return LidarPointCloud.from_buffer(blob, file_type) if blob is not None else None


def load_image(image: ImageDBRow.Image, local_store: LocalStore, remote_store: S3Store) -> Optional[Image]:
    """
    Loads an image given a database Image object.
    :param image: The image for which to grab the image.
    :param local_store: Local blob store for loading blobs from local file system.
    :param remote_store: S3 blob store for loading blobs from AWS S3.
    :return: The corresponding image.
    """
    blob = download_and_cache(image.filename_jpg, local_store, remote_store)
    return Image.from_buffer(blob) if blob is not None else None
