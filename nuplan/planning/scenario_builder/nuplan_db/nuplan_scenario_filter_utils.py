from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.database.nuplan_db.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.nuplan_db.scene import Scene
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
    ScenarioMapping,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map

logger = logging.getLogger(__name__)

# Dictionary that holds a list of scenarios for each scenario type
ScenarioDict = Dict[str, List[NuPlanScenario]]

# Scene indices smaller that the first valid index or larger than the last valid index are dropped during filtering.
# This is done to ensure that all selected scenes have at least 20s of history/future samples.
FIRST_VALID_SCENE_IDX = 2  # First scene in a log that is considered valid for training/simulation
LAST_VALID_SCENE_IDX = -2  # Last scene in a log that is considered valid for training/simulation


@dataclass(frozen=True)
class FilterWrapper:
    """
    Generic filter wrapper that encapsulates the filter's function and metadata.
    """

    fn: Callable[[ScenarioDict], ScenarioDict]  # function that filters the scenario dictionary
    enable: bool  # whether to run this filter
    name: str  # name of the filter

    def run(self, scenario_dict: ScenarioDict) -> ScenarioDict:
        """
        Run the filter if enabled.
        :param scenario_dict: Input scenario dictionary.
        :return: Output scenario dictionary.
        """
        if not self.enable:
            return scenario_dict

        logger.debug(f'Running scenario filter {self.name}...')
        scenario_dict = self.fn(scenario_dict)  # type: ignore
        logger.debug(f'Running scenario filter {self.name}...DONE')

        return scenario_dict


def is_scene_valid(
    scene: Scene, first_valid_idx: int = FIRST_VALID_SCENE_IDX, last_valid_idx: int = LAST_VALID_SCENE_IDX
) -> bool:
    """
    Check whether the scene has enough history/future buffer and is valid for training/simulation.
    :param scene: Candidate scene.
    :param first_valid_idx: Index of first valid scene.
    :param last_valid_idx: Index of last valid scene.
    :return: Whether the scene is valid or not.
    """
    scenes = scene.log.scenes
    scene_idx = int(scenes.index(scene))
    return first_valid_idx <= scene_idx < len(scenes) + last_valid_idx


def extract_scenes_from_log_db(
    db: NuPlanDB, first_valid_idx: int = FIRST_VALID_SCENE_IDX, last_valid_idx: int = LAST_VALID_SCENE_IDX
) -> List[Scene]:
    """
    Retrieve all valid scenes from a log database.
    :param db: Log database to retrieve scenes from.
    :param first_valid_idx: Index of first valid scene.
    :param last_valid_idx: Index of last valid scene.
    :return: Retrieved scenes.
    """
    return list(db.scene)[first_valid_idx:last_valid_idx]


def create_scenarios_by_tokens(
    scenario_tokens: List[Tuple[str, str]],
    db: NuPlanDBWrapper,
    log_names: Optional[List[str]],
    expand_scenarios: bool,
    vehicle_parameters: VehicleParameters,
    ground_truth_predictions: Optional[TrajectorySampling],
) -> ScenarioDict:
    """
    Create initial scenario dictionary based on desired tokens.
    :param scenario_tokens: List of (log_name, lidarpc_tokens) used to initialize the scenario dict.
    :param db: Object for accessing the available databases.
    :param log_names: List of log names to include in the scenario dictionary.
    :param expand_scenarios: Whether to expand multi-sample scenarios to multiple single-sample scenarios.
    :param vehicle_parameters: Vehicle parameters for this db.
    :param ground_truth_predictions: If None, no GT predictions will be extracted based on its future setting.
    :return: Dictionary that holds a list of scenarios for each scenario type.
    """
    logger.debug("Creating scenarios by tokens...")

    # Whether to expand scenarios from multi-sample to single-sample scenarios
    extraction_info = None if expand_scenarios else ScenarioExtractionInfo()

    # Find all tokens that match the desired log names
    if log_names:
        candidate_log_names = set(log_names)
        scenario_tokens = [(log_name, token) for log_name, token in scenario_tokens if log_name in candidate_log_names]

    # Construct nuplan scenario objects for each (log_name, lidarpc token) pair
    args = [DEFAULT_SCENARIO_NAME, extraction_info, vehicle_parameters, ground_truth_predictions]
    scenarios = [NuPlanScenario(db.get_log_db(log_name), log_name, token, *args) for log_name, token in scenario_tokens]

    return {DEFAULT_SCENARIO_NAME: scenarios}


def create_scenarios_by_types(
    scenario_types: List[str],
    db: NuPlanDBWrapper,
    log_names: Optional[List[str]],
    expand_scenarios: bool,
    scenario_mapping: ScenarioMapping,
    vehicle_parameters: VehicleParameters,
    ground_truth_predictions: Optional[TrajectorySampling],
) -> ScenarioDict:
    """
    Create initial scenario dictionary based on desired scenario types.
    :param scenario_types: List of scenario types used to filter the pool of scenarios.
    :param db: Object for accessing the available databases.
    :param log_names: List of log names to include in the scenario dictionary.
    :param expand_scenarios: Whether to expand multi-sample scenarios to multiple single-sample scenarios.
    :param vehicle_parameters: Vehicle parameters for this db.
    :param ground_truth_predictions: If None, no GT predictions will be extracted based on its future setting.
    :return: Dictionary that holds a list of scenarios for each scenario type.
    """
    logger.debug(f"Creating scenarios by types {scenario_types}...")

    # Dictionary that holds a list of scenarios for each scenario type
    scenario_dict: ScenarioDict = dict()

    # Find all candidate scenario types
    available_types = db.get_all_scenario_types()
    candidate_types = set(scenario_types).intersection(available_types)

    # Find all log dbs that match the desired log names
    log_dbs = db.log_dbs
    if log_names:
        candidate_log_names = set(log_names)
        log_dbs = [log_db for log_db in log_dbs if log_db.name in candidate_log_names]

    # Populate scenario dictionary with list of scenarios for each type
    for scenario_type in candidate_types:
        extraction_info = None if expand_scenarios else scenario_mapping.get_extraction_info(scenario_type)

        # TODO: Make scenario_tag.select_many method in DB
        args = [scenario_type, extraction_info, vehicle_parameters, ground_truth_predictions]
        scenario_dict[scenario_type] = [
            NuPlanScenario(log_db, log_db.log_name, tag.lidar_pc_token, *args)
            for log_db in log_dbs
            for tag in log_db.scenario_tag.select_many(type=scenario_type)
            if is_scene_valid(tag.lidar_pc.scene)
        ]

    return scenario_dict


def create_all_scenarios(
    db: NuPlanDBWrapper,
    log_names: Optional[List[str]],
    expand_scenarios: bool,
    vehicle_parameters: VehicleParameters,
    worker: WorkerPool,
    ground_truth_predictions: Optional[TrajectorySampling],
) -> ScenarioDict:
    """
    Create initial scenario dictionary containing all available scenarios in the scenario pool.
    :param db: Object for accessing the available databases.
    :param log_names: List of log names to include in the scenario dictionary.
    :param expand_scenarios: Whether to expand multi-sample scenarios to multiple single-sample scenarios.
    :param vehicle_parameters: Vehicle parameters for this db.
    :param worker: Worker pool for concurrent scenario processing.
    :param ground_truth_predictions: If None, no GT predictions will be extracted based on its future setting
    :return: Dictionary that holds a list of scenarios for each scenario type.
    """
    logger.debug('Creating all scenarios...')

    # Whether to expand scenarios from multi-sample to single-sample scenarios
    extraction_info = None if expand_scenarios else ScenarioExtractionInfo()

    def get_scenarios_from_log_dbs(log_dbs: List[NuPlanDB]) -> List[NuPlanScenario]:
        """
        Retrieve a list of nuplan scenario objects from a list of nuplan log databases.
        :param log_db: List of nuplan log databases.
        :return: List of nuplan scenarios.
        """

        def get_scenarios_from_log_db(log_db: NuPlanDB) -> List[NuPlanScenario]:
            """
            Retrieve a list of nuplan scenario objects from a single nuplan log database.
            Note: This method uses variables from the outer scope to avoid transferring unnecessary load across workers.
            :param log_db: Nuplan log database.
            :return: List of nuplan scenarios.
            """
            # Total list of scene tokens in the database
            scene_tokens = [scene.token for scene in extract_scenes_from_log_db(log_db)]

            query = (
                log_db.session.query(LidarPc.token)
                .filter(LidarPc.scene_token.in_(scene_tokens))
                .order_by(LidarPc.timestamp.asc())
                .all()
            )

            # Construct nuplan scenario objects for this log
            args = [DEFAULT_SCENARIO_NAME, extraction_info, vehicle_parameters, ground_truth_predictions]
            scenarios = [NuPlanScenario(log_db, log_db.log_name, token, *args) for token, in query]

            return scenarios

        return [scenario for log_db in log_dbs for scenario in get_scenarios_from_log_db(log_db)]

    # Find all log dbs that match the desired log names
    log_dbs = db.log_dbs
    if log_names:
        candidate_log_names = set(log_names)
        log_dbs = [log_db for log_db in log_dbs if log_db.name in candidate_log_names]

    # Retrieve all scenarios for the total list of scenes concurrently
    scenarios = worker_map(worker, get_scenarios_from_log_dbs, log_dbs)

    return {DEFAULT_SCENARIO_NAME: scenarios}


def filter_by_log_names(scenario_dict: ScenarioDict, log_names: List[str]) -> ScenarioDict:
    """
    Filter a scenario dictionary by log names.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param log_names: List of log names to include in the scenario dictionary.
    :return: Filtered scenario dictionary.
    """
    scenario_dict = {
        scenario_type: [scenario for scenario in scenarios if scenario.log_name in log_names]
        for scenario_type, scenarios in scenario_dict.items()
    }

    return scenario_dict


def filter_by_map_names(scenario_dict: ScenarioDict, map_names: List[str], db: NuPlanDBWrapper) -> ScenarioDict:
    """
    Filter a scenario dictionary by map names.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param map_names: List of map names to include in the scenario dictionary.
    :param db: Object for accessing the available log databases.
    :return: Filtered scenario dictionary.
    """
    # Mapping from log name to map version
    # TODO: Pass map name in scenario
    log_maps = {log_db.log_name: log_db.map_name for log_db in db.log_dbs}

    scenario_dict = {
        scenario_type: [scenario for scenario in scenarios if log_maps[scenario.log_name] in map_names]
        for scenario_type, scenarios in scenario_dict.items()
    }

    return scenario_dict


def filter_num_scenarios_per_type(
    scenario_dict: ScenarioDict, num_scenarios_per_type: int, randomize: bool
) -> ScenarioDict:
    """
    Filter the number of scenarios in a scenario dictionary per scenario type.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param num_scenarios_per_type: Number of scenarios per type to keep.
    :param randomize: Whether to randomly sample the scenarios.
    :return: Filtered scenario dictionary.
    """
    for scenario_type in scenario_dict:
        if randomize and num_scenarios_per_type < len(scenario_dict[scenario_type]):  # Sample scenarios randomly
            scenario_dict[scenario_type] = random.sample(scenario_dict[scenario_type], num_scenarios_per_type)
        else:  # Sample the top k number of scenarios per type
            scenario_dict[scenario_type] = scenario_dict[scenario_type][:num_scenarios_per_type]

    return scenario_dict


def filter_total_num_scenarios(
    scenario_dict: ScenarioDict, limit_total_scenarios: Union[int, float], randomize: bool
) -> ScenarioDict:
    """
    Filter the total number of scenarios in a scenario dictionary.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param limit_total_scenarios: Number of total scenarios to keep.
    :param randomize: Whether to randomly sample the scenarios.
    :return: Filtered scenario dictionary.
    """
    scenario_list = scenario_dict_to_list(scenario_dict)

    if isinstance(limit_total_scenarios, int):  # Exact number of scenarios to keep
        max_scenarios = limit_total_scenarios
        scenario_list = (
            random.sample(scenario_list, max_scenarios)
            if randomize and max_scenarios < len(scenario_list)
            else scenario_list[:max_scenarios]
        )
    elif isinstance(limit_total_scenarios, float):  # Percentage of scenarios to keep
        sample_ratio = limit_total_scenarios
        assert 0.0 < sample_ratio < 1.0, f'Sample ratio has to be between 0 and 1, got {sample_ratio}'
        step = int(1.0 / sample_ratio)
        if step < len(scenario_list):
            scenario_list = scenario_list[::step]
    else:
        raise TypeError('Scenario filter "limit_total_scenarios" must be of type int or float')

    return scenario_list_to_dict(scenario_list)


def filter_invalid_goals(scenario_dict: ScenarioDict, worker: WorkerPool) -> ScenarioDict:
    """
    Filter the scenarios with invalid mission goals in a scenario dictionary.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param worker: Worker pool for concurrent scenario processing.
    :return: Filtered scenario dictionary.
    """

    def _filter_goals(scenarios: List[NuPlanScenario]) -> List[NuPlanScenario]:
        """
        Filter scenarios that contain invalid mission goals.
        :param scenarios: List of scenarios to filter.
        :return: List of filtered scenarios.
        """
        return [scenario for scenario in scenarios if scenario.get_mission_goal()]

    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = worker_map(worker, _filter_goals, scenario_dict[scenario_type])

    return scenario_dict


def scenario_dict_to_list(scenario_dict: ScenarioDict, shuffle: Optional[bool] = None) -> List[NuPlanScenario]:
    """
    Unravel a scenario dictionary to a list of scenarios.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param shuffle: Whether to shuffle the scenario list.
    :return: List of scenarios.
    """
    scenarios = [scenario for scenario_list in scenario_dict.values() for scenario in scenario_list]
    scenarios = sorted(scenarios, key=lambda scenario: scenario.token)  # type: ignore

    if shuffle:
        random.shuffle(scenarios)

    return scenarios


def scenario_list_to_dict(scenario_list: List[NuPlanScenario]) -> ScenarioDict:
    """
    Convert a list of scenarios to a dictionary.
    :param scenario_list: List of input scenarios.
    :return: Dictionary that holds a list of scenarios for each scenario type.
    """
    scenario_dict: ScenarioDict = defaultdict(list)

    for scenario in scenario_list:
        scenario_dict[scenario.scenario_type].append(scenario)

    return scenario_dict
