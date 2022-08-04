from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.utils.s3_utils import check_s3_path_exists, expand_s3_dir
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioMapping,
    download_file_if_necessary,
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


@dataclass(frozen=True)
class GetScenariosFromDbFileParams:
    """
    A convenience class for holding all the parameters to get_scenarios_from_log_file
    """

    # The root folder for the db file (e.g. "/data/sets/nuplan")
    data_root: str

    # The absolute path log file to query
    # e.g. /data/sets/nuplan-v1.0/mini/2021.10.11.08.31.07_veh-50_01750_01948.db
    log_file_absolute_path: str

    # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    expand_scenarios: bool

    # The root directory for maps (e.g. "/data/sets/nuplan/maps")
    map_root: str

    # The map version to load (e.g. "1.0")
    map_version: str

    # The ScenarioMapping to pass to the constructed scenarios.
    scenario_mapping: ScenarioMapping

    # The ego vehicle parameters to pass to the constructed scenarios.
    vehicle_parameters: VehicleParameters

    # The ground_truth_prediction sampling parameters to pass to the constructed scenarios.
    ground_truth_predictions: Optional[TrajectorySampling]

    # If provided, the tokens on which to filter.
    filter_tokens: Optional[List[str]]

    # If provided, the scenario types on which to filter.
    filter_types: Optional[List[str]]

    # If provided, the map names on which to filter (e.g. "[us-nv-las-vegas-strip, us-ma-boston]")
    filter_map_names: Optional[List[str]]


def get_db_filenames_from_load_path(load_path: str) -> List[str]:
    """
    Retrieve all log database filenames from a load path.
    The path can be either local or remote (S3).
    The path can represent either a single database filename (.db file) or a directory containing files.
    :param load_path: Load path, it can be a filename or list of filenames.
    :return: A list of all discovered log database filenames.
    """
    if load_path.endswith('.db'):  # Single database path
        if load_path.startswith('s3://'):  # File is remote (S3)
            assert check_s3_path_exists(load_path), f'S3 db path does not exist: {load_path}'
            os.environ['NUPLAN_DATA_ROOT_S3_URL'] = load_path.rstrip(Path(load_path).name)
        else:  # File is local
            assert Path(load_path).is_file(), f'Local db path does not exist: {load_path}'
        db_filenames = [load_path]
    else:  # Path to directory containing databases
        if load_path.startswith('s3://'):  # Directory is remote (S3)
            db_filenames = expand_s3_dir(load_path, filter_suffix='.db')
            assert len(db_filenames) > 0, f'S3 dir does not contain any dbs: {load_path}'
            os.environ['NUPLAN_DATA_ROOT_S3_URL'] = load_path  # TODO: Deprecate S3 data root env variable
        elif Path(load_path).expanduser().is_dir():  # Directory is local
            db_filenames = [
                str(path) for path in sorted(Path(load_path).expanduser().iterdir()) if path.suffix == '.db'
            ]
        else:
            raise ValueError(f'Expected db load path to be file, dir or list of files/dirs, but got {load_path}')

    return db_filenames


def discover_log_dbs(load_path: Union[List[str], str]) -> List[str]:
    """
    Discover all log dbs from the input load path.
    If the path is a filename, expand the path and return the list of filenames in that path.
    Else, if the path is already a list, expand each path in the list and return the flattened list.
    :param load_path: Load path, it can be a filename or list of filenames of a database and/or dirs of databases.
    :return: A list with all discovered log database filenames.
    """
    if isinstance(load_path, list):  # List of database paths
        nested_db_filenames = [get_db_filenames_from_load_path(path) for path in sorted(load_path)]
        db_filenames = [filename for filenames in nested_db_filenames for filename in filenames]
    else:
        db_filenames = get_db_filenames_from_load_path(load_path)

    return db_filenames


def get_scenarios_from_db_file(params: GetScenariosFromDbFileParams) -> ScenarioDict:
    """
    Gets all of the scenarios present in a single sqlite db file that match the provided filter parameters.
    :param params: The filter parameters to use.
    :return: A ScenarioDict containing the relevant scenarios.
    """
    local_log_file_absolute_path = download_file_if_necessary(params.data_root, params.log_file_absolute_path)

    scenario_dict: ScenarioDict = {}
    for row in get_scenarios_from_db(
        local_log_file_absolute_path, params.filter_tokens, params.filter_types, params.filter_map_names
    ):
        scenario_type = row["scenario_type"]

        if scenario_type is None:
            scenario_type = DEFAULT_SCENARIO_NAME

        if scenario_type not in scenario_dict:
            scenario_dict[scenario_type] = []

        extraction_info = (
            None if params.expand_scenarios else params.scenario_mapping.get_extraction_info(scenario_type)
        )

        scenario_dict[scenario_type].append(
            NuPlanScenario(
                params.data_root,
                params.log_file_absolute_path,
                row["token"].hex(),
                row["timestamp"],
                scenario_type,
                params.map_root,
                params.map_version,
                row["map_name"],
                extraction_info,
                params.vehicle_parameters,
                params.ground_truth_predictions,
            )
        )

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
