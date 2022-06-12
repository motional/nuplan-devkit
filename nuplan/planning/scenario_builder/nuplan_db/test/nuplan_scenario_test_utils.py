from __future__ import annotations

from functools import lru_cache
from typing import Optional

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.tests.nuplan_db_test_utils import (
    NUPLAN_DATA_ROOT,
    NUPLAN_DB_FILES,
    NUPLAN_MAP_VERSION,
    NUPLAN_MAPS_ROOT,
    get_test_nuplan_db,
    get_test_nuplan_lidarpc,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)


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
    :param lidar_pc_index: If available, lidar pc with defined index will be used
    :return: A sample db scenario.
    """
    db = get_test_nuplan_db()
    lidarpc = get_test_nuplan_lidarpc() if lidar_pc_index is None else get_test_nuplan_db().lidar_pc[lidar_pc_index]

    scenario_extraction_info = ScenarioExtractionInfo() if use_multi_sample else None

    scenario = NuPlanScenario(
        db=db,
        log_name=lidarpc.log.logfile,
        initial_lidar_token=lidarpc.token,
        scenario_extraction_info=scenario_extraction_info,
        scenario_type=DEFAULT_SCENARIO_NAME,
        ego_vehicle_parameters=get_pacifica_parameters(),
    )

    return scenario
