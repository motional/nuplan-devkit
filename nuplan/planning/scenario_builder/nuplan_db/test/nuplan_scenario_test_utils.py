from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_sensor_data_token_timestamp_from_db,
    get_sensor_token_by_index_from_db,
    get_sensor_token_map_name_from_db,
)
from nuplan.database.tests.test_utils_nuplan_db import NUPLAN_DB_FILES
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)

NUPLAN_DATA_ROOT = os.environ["NUPLAN_DATA_ROOT"]
NUPLAN_SENSOR_ROOT = os.path.join(NUPLAN_DATA_ROOT, 'sensor_blobs')
NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]
NUPLAN_MAP_VERSION = os.environ["NUPLAN_MAP_VERSION"]

# Chosen arbitrarily to retain backward compatibility
DEFAULT_LIDARPC_INDEX = 1000


@lru_cache(maxsize=1)
def get_test_nuplan_scenario_builder() -> NuPlanScenarioBuilder:
    """Get a nuPlan scenario builder object with default settings to be used in testing."""
    return NuPlanScenarioBuilder(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        sensor_root=NUPLAN_SENSOR_ROOT,
        db_files=NUPLAN_DB_FILES,
        map_version=NUPLAN_MAP_VERSION,
    )


@lru_cache(maxsize=1)
def get_test_nuplan_scenario(
    use_multi_sample: bool = False, lidar_pc_index: Optional[int] = None, sensor_root: Optional[str] = None
) -> NuPlanScenario:
    """
    Retrieve a sample scenario from the db.
    :param use_multi_sample: Whether to extract multiple temporal samples in the scenario.
    :param lidar_pc_index: The initial lidarpc_token for the sceanrio. If None, then a default example (corresponding to DEFAULT_LIDARPC_INDEX) will be used.
    :param sensor_root: The directory for which the sensor data should be saved to.
    :return: A sample db scenario.
    """
    # This file chosen to maintain backwards compatibility with existing tests.
    load_path = NUPLAN_DB_FILES[4]

    lidar_pc_index = DEFAULT_LIDARPC_INDEX if lidar_pc_index is None else lidar_pc_index

    token = get_sensor_token_by_index_from_db(load_path, get_lidarpc_sensor_data(), lidar_pc_index)
    timestamp = get_sensor_data_token_timestamp_from_db(load_path, get_lidarpc_sensor_data(), token)
    map_name = get_sensor_token_map_name_from_db(load_path, get_lidarpc_sensor_data(), token)

    if timestamp is None or map_name is None:
        raise ValueError(f"Token {token} not found in log.")

    scenario = NuPlanScenario(
        data_root=NUPLAN_DATA_ROOT,
        log_file_load_path=load_path,
        initial_lidar_token=token,
        initial_lidar_timestamp=timestamp,
        scenario_type=DEFAULT_SCENARIO_NAME,
        map_root=NUPLAN_MAPS_ROOT,
        map_version=NUPLAN_MAP_VERSION,
        map_name=map_name,
        scenario_extraction_info=ScenarioExtractionInfo() if use_multi_sample else None,
        ego_vehicle_parameters=get_pacifica_parameters(),
        sensor_root=NUPLAN_SENSOR_ROOT if sensor_root is None else sensor_root,
    )

    return scenario
