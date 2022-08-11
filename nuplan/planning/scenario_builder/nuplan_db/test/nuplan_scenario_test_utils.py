from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_lidarpc_token_by_index_from_db,
    get_lidarpc_token_map_name_from_db,
    get_lidarpc_token_timestamp_from_db,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)

NUPLAN_DATA_ROOT = os.environ["NUPLAN_DATA_ROOT"]
NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]
NUPLAN_MAP_VERSION = os.environ["NUPLAN_MAP_VERSION"]

# Chosen arbitrarily to retain backward compatibility
DEFAULT_LIDARPC_INDEX = 1000

# Use a subset of the mini split for testing purposes
NUPLAN_DB_FILES = [
    # Mini train
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.07.16.20.45.29_veh-35_01095_01486.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.09.15.16.17.26_veh-28_00151_00569.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.08.17.18.54.02_veh-45_00665_01065.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.10.11.02.57.41_veh-50_01522_02088.db",
    # Mini val
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.06.08.16.31.33_veh-38_01589_02072.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.08.31.14.40.58_veh-40_00285_00668.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.08.24.13.12.55_veh-45_00386_00867.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.10.05.07.10.04_veh-52_01442_01802.db",
    # Mini test
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.06.28.16.57.59_veh-26_00016_00484.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.08.30.14.54.34_veh-40_00439_00835.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.09.16.15.12.03_veh-42_01037_01434.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.10.06.07.26.10_veh-52_00006_00398.db",
]


@lru_cache(maxsize=1)
def get_test_nuplan_scenario_builder() -> NuPlanScenarioBuilder:
    """Get a nuPlan scenario builder object with default settings to be used in testing."""
    return NuPlanScenarioBuilder(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        db_files=NUPLAN_DB_FILES,
        map_version=NUPLAN_MAP_VERSION,
    )


@lru_cache(maxsize=1)
def get_test_nuplan_scenario(use_multi_sample: bool = False, lidar_pc_index: Optional[int] = None) -> NuPlanScenario:
    """
    Retrieve a sample scenario from the db.
    :param use_multi_sample: Whether to extract multiple temporal samples in the scenario.
    :param lidar_pc_index: The initial lidarpc_token for the sceanrio. If None, then a default example (corresponding to DEFAULT_LIDARPC_INDEX) will be used.
    :return: A sample db scenario.
    """
    # This file chosen to maintain backwards compatibility with existing tests.
    load_path = NUPLAN_DB_FILES[4]

    lidar_pc_index = DEFAULT_LIDARPC_INDEX if lidar_pc_index is None else lidar_pc_index

    token = get_lidarpc_token_by_index_from_db(load_path, lidar_pc_index)
    timestamp = get_lidarpc_token_timestamp_from_db(load_path, token)
    map_name = get_lidarpc_token_map_name_from_db(load_path, token)

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
    )

    return scenario
