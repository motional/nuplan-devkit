from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union, cast

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.utils.s3_utils import check_s3_path_exists, expand_s3_dir
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioMapping,
    download_file_if_necessary,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import get_neighbor_vector_map
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
    # e.g. /data/sets/nuplan-v1.1/splits/mini/2021.10.11.08.31.07_veh-50_01750_01948.db
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

    # If provided, the tokens on which to filter.
    filter_tokens: Optional[List[str]]

    # If provided, the scenario types on which to filter.
    filter_types: Optional[List[str]]

    # If provided, the map names on which to filter (e.g. "[us-nv-las-vegas-strip, us-ma-boston]")
    filter_map_names: Optional[List[str]]

    # The root directory for sensor blobs (e.g. "/data/sets/nuplan/nuplan-v1.1/sensor_blobs")
    sensor_root: str

    # If provided, whether to remove scenarios without a valid mission goal.
    remove_invalid_goals: bool = False

    # If provided, the scenario will contain camera data on the anchor lidar_pc.
    include_cameras: bool = False

    # Verbosity, provides download progression
    verbose: bool = False


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
        nested_db_filenames = [get_db_filenames_from_load_path(path) for path in sorted(set(load_path))]
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
    local_log_file_absolute_path = download_file_if_necessary(
        params.data_root, params.log_file_absolute_path, params.verbose
    )

    scenario_dict: ScenarioDict = {}
    for row in get_scenarios_from_db(
        local_log_file_absolute_path,
        params.filter_tokens,
        params.filter_types,
        params.filter_map_names,
        not params.remove_invalid_goals,
        params.include_cameras,
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
                data_root=params.data_root,
                log_file_load_path=params.log_file_absolute_path,
                initial_lidar_token=row["token"].hex(),
                initial_lidar_timestamp=row["timestamp"],
                scenario_type=scenario_type,
                map_root=params.map_root,
                map_version=params.map_version,
                map_name=row["map_name"],
                scenario_extraction_info=extraction_info,
                ego_vehicle_parameters=params.vehicle_parameters,
                sensor_root=params.sensor_root,
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
        else:  # Do equisampling for each scenario type
            step = max(len(scenario_dict[scenario_type]) // num_scenarios_per_type, 1)
            scenario_dict[scenario_type] = scenario_dict[scenario_type][::step]
            # In the case that floor division results in more samples than desired, truncate the list
            scenario_dict[scenario_type] = scenario_dict[scenario_type][:num_scenarios_per_type]

    return scenario_dict


# TODO: Move to SQL layer
def filter_scenarios_by_timestamp(
    scenario_dict: ScenarioDict,
    timestamp_threshold_s: float = 5.0,
) -> ScenarioDict:
    """
    Filter the scenarios in a scenario dictionary by timestamp. Scenarios that occur are within `timestamp_threshold` of a particular scenario will be removed.
    This is only to be used during caching or during simulation. This currently cannot be used during training as `CachedScenario` does not implement timestamp information.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param timestamp_threshold_s: Threshold for filtering out scenarios clustered together in time.
    :return: Filtered scenario dictinoary.
    """
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = _filter_scenarios_by_timestamp(
            scenario_dict[scenario_type], timestamp_threshold_s
        )

    return scenario_dict


# TODO: Move to SQL layer
def _filter_scenarios_by_timestamp(
    scenario_list: List[NuPlanScenario], timestamp_threshold_s: float
) -> List[NuPlanScenario]:
    """
    Filters the list of scenarios by timestamp.
    :param scenario_list: List of scenarios to filtered.
    :param timestamp_threshold_s: Threshold for filtering out scenarios clustered together in time.
    :return: Filtered list of scenarios.
    """
    if len(scenario_list) == 0:
        return scenario_list

    def _extract_initial_lidar_timestamp(scenario: NuPlanScenario) -> int:
        return cast(int, scenario._initial_lidar_timestamp)

    scenario_list.sort(key=_extract_initial_lidar_timestamp)
    filtered_scenarios = []
    min_next_timestamp = scenario_list[0]._initial_lidar_timestamp * 1e-6  # convert to seconds

    for scenario in scenario_list:
        if scenario._initial_lidar_timestamp * 1e-6 >= min_next_timestamp:
            filtered_scenarios.append(scenario)
            min_next_timestamp = scenario._initial_lidar_timestamp * 1e-6 + timestamp_threshold_s

    return filtered_scenarios


def filter_total_num_scenarios(
    scenario_dict: ScenarioDict, limit_total_scenarios: Union[int, float], randomize: bool
) -> ScenarioDict:
    """
    Filter the total number of scenarios in a scenario dictionary to reach a certain percentage of
    the original dataset (eg. 10% or 0.1 of the original dataset) or a fixed number of scenarios (eg. 100 scenarios).

    In the scenario dataset, a small proportion of the scenarios are labelled with a scenario_type
    (eg. stationary, following_lane_with_lead, etc). These labelled scenarios are snapshots in time,
    labelled at regular intervals. The rest of the timesteps in between these labelled scenarios are
    the unlabelled scenario types which are given a scenario_type of DEFAULT_SCENARIO_NAME ('unknown'),
    making up a majority of the dataset.
    This function filters the scenarios while preserving as much of the labelled scenarios as possible
    by removing the unlabelled scenarios first, followed by the labelled scenarios if necessary.

    Example:
    Original dataset = 100 scenarios (90 unknown/5 stationary/5 following_lane_with_lead)
    Setting limit_total_scenarios = 0.5 during caching => 50 scenarios in cache (40 unknown/5 stationary/5 following_lane_with_lead)
    Setting limit_total_scenarios = 0.1 during caching => 10 scenarios in cache (5 stationary/5 following_lane_with_lead)
    Setting limit_total_scnearios = 0.02 during caching => 2 scenarios in cache (1 stationary/1 following_lane_with_lead)

    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param limit_total_scenarios: Number of total scenarios to keep.
    :param randomize: Whether to randomly sample the scenarios.
    :return: Filtered scenario dictionary.
    """
    total_num_scenarios = sum(len(scenarios) for scenarios in scenario_dict.values())

    if isinstance(limit_total_scenarios, int):  # Exact number of scenarios to keep
        assert (
            limit_total_scenarios > 0
        ), "Number of samples kept should be more than 0 in order to not have an empty cache."
        num_nodes = int(os.environ.get("NUM_NODES", 1))
        required_num_scenarios = math.ceil(limit_total_scenarios / num_nodes)  # Get number of scenarios to keep
        # Only remove scenarios if the limit is less than the total number of scenarios
        if required_num_scenarios < total_num_scenarios:
            scenario_dict = _filter_scenarios(scenario_dict, total_num_scenarios, required_num_scenarios, randomize)

    elif isinstance(limit_total_scenarios, float):  # Percentage of scenarios to keep
        sample_ratio = limit_total_scenarios
        assert 0.0 < sample_ratio < 1.0, f'Sample ratio has to be between 0 and 1, got {sample_ratio}'
        required_num_scenarios = math.ceil(sample_ratio * total_num_scenarios)  # Get number of scenarios to keep

        scenario_dict = _filter_scenarios(scenario_dict, total_num_scenarios, required_num_scenarios, randomize)

    else:
        raise TypeError('Scenario filter "limit_total_scenarios" must be of type int or float')

    return scenario_dict


def _filter_scenarios(
    scenario_dict: ScenarioDict, total_num_scenarios: int, required_num_scenarios: int, randomize: bool
) -> ScenarioDict:
    """
    Filters scenarios until we reach the user specified number of scenarios. Scenarios with scenario_type DEFAULT_SCENARIO_NAME are removed first either randomly or with equisampling, and subsequently
    the other scenarios are sampled randomly or with equisampling if necessary.
    :param scenario_dict: Dictionary containining a mapping of scenario_type to a list of the AbstractScenario objects.
    :param total_num_scenarios: Total number of scenarios in the scenario dictionary.
    :param required_num_scenarios: Number of scenarios desired.
    :param randomize: boolean to decide whether to randomize the sampling of scenarios.
    :return: Scenario dictionary with the required number of scenarios.
    """

    def _filter_scenarios_from_scenario_list(
        scenario_list: List[NuPlanScenario], num_scenarios_to_keep: int, randomize: bool
    ) -> List[NuPlanScenario]:
        """
        Removes scenarios randomly or does equisampling of the scenarios.
        :param scenario_list: List of scenarios.
        :param num_scenarios_to_keep: Number of scenarios that should be in the final list.
        :param randomize: Boolean for whether to randomly sample from scenario_list or carry out equisampling of scenarios.
        """
        total_num_scenarios = len(scenario_list)
        step = max(total_num_scenarios // num_scenarios_to_keep, 1)
        scenario_list = random.sample(scenario_list, num_scenarios_to_keep) if randomize else scenario_list[::step]
        # In the case that floor division results in more samples than desired, truncate the list
        scenario_list = scenario_list[:num_scenarios_to_keep]

        return scenario_list

    if total_num_scenarios == 0 or required_num_scenarios == 0 or len(scenario_dict) == 0:
        return {}

    if DEFAULT_SCENARIO_NAME in scenario_dict:
        num_default_scenarios = len(scenario_dict[DEFAULT_SCENARIO_NAME])

        # if we can reach the desired number of scenarios by removing the default scenarios, we will not remove any known scenario types
        if total_num_scenarios - required_num_scenarios < num_default_scenarios:
            num_default_scenarios_to_keep = num_default_scenarios - (total_num_scenarios - required_num_scenarios)
            scenario_dict[DEFAULT_SCENARIO_NAME] = _filter_scenarios_from_scenario_list(
                scenario_dict[DEFAULT_SCENARIO_NAME], num_default_scenarios_to_keep, randomize
            )
            return scenario_dict
        else:  # if removing default scenarios is insufficient, remove all DEFAULT_SCENARIO_NAME scenarios first then remove from the known scenario types
            scenario_dict.pop(DEFAULT_SCENARIO_NAME)

    scenario_list = scenario_dict_to_list(scenario_dict)
    scenario_list = _filter_scenarios_from_scenario_list(scenario_list, required_num_scenarios, randomize)
    scenario_dict = scenario_list_to_dict(scenario_list)

    return scenario_dict


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


def filter_fraction_lidarpc_tokens_in_set(
    scenario_dict: ScenarioDict, token_set_path: Path, fraction_threshold: float
) -> ScenarioDict:
    """
    Filter out all scenarios from a nuplan ScenarioDict for whom the fraction of the scenario's lidarpc tokens
        in token_set is less than or equal to fraction_threshold (strictly less for fraction_threshold=1).
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param token_set_path: a path to List of lidarpc tokens from a Nuplan DB, stored as json.
    :param fraction_threshold: a float in [0, 1].
    :return: a Dictionary with the same structure as scenario dict, but in which all individual scenarios
        for whom the fraction of its tokens that are contained in token set is <= fraction_threshold
        (or < fraction_threshold if fraction_threshold is 1)
    """
    if not 0 <= fraction_threshold <= 1:
        raise ValueError("Fraction_threshold must be in [0,1].")

    with open(token_set_path, "r") as token_file:
        token_list = json.load(token_file)
        if type(token_list) != list or type(token_list[0]) != str:
            raise ValueError("token_set_path does not point to a json-formatted list of strings.")
        token_set = set(token_list)

    def _are_lidarpc_tokens_in_set(scenario: NuPlanScenario, token_set: Set[str], fraction_threshold: float) -> bool:
        """
        For a single scenario, report whether (True/False) the fraction of the scenario's lidarpc tokens
            in token_set is greater than fraction_threshold (greater than or equal to for fraction_threshold=1).
        :param scenario: a valid NuplanScenario instance.
        :param token_set: a Pyton Set of lidarpc tokens from a Nuplan DB.
        :param fraction_threshold: a Python float in [0, 1].
        :return: True if strictly more than fraction_threshold fraction of the lidarpc tokens in scenario belong to
            token_set (strictly equal if fraction_threshold is 1)
        """
        scenario_tokens = set(scenario.get_scenario_tokens())

        if fraction_threshold == 1:
            return scenario_tokens == token_set

        return len(scenario_tokens.intersection(token_set)) / len(scenario_tokens) > fraction_threshold

    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = list(
            filter(
                lambda scenario: _are_lidarpc_tokens_in_set(scenario, token_set, fraction_threshold),
                scenario_dict[scenario_type],
            )
        )

    return scenario_dict


def _is_non_stationary(scenario: NuPlanScenario, minimum_threshold: float) -> bool:
    """
    Determines whether the ego cumulatively moves at least minimum_threshold meters over the course of a given scenario
    :param scenario: a NuPlan expert scenario
    :param minimum_threshold: minimum distance in meters (inclusive) the ego center has to travel in the scenario
        for the ego to be determined non-stationary
    :return: True if the cumulative frame-to-frame displacement of the ego center in the scenario
        is >= the minimum threshold
    """
    trajectory = scenario.get_ego_future_trajectory(iteration=0, time_horizon=scenario.duration_s.time_s)
    trajectory_xy_matrix = np.array([[state.center.x, state.center.y] for state in trajectory])  # type: ignore
    current_state = trajectory_xy_matrix[:-1]
    next_state = trajectory_xy_matrix[1:]
    total_ego_displacement = np.sum(np.linalg.norm(next_state - current_state, axis=1))
    return bool(total_ego_displacement >= minimum_threshold)


def filter_non_stationary_ego(scenario_dict: ScenarioDict, minimum_threshold: float) -> ScenarioDict:
    """
    Filters a ScenarioDict, leaving only scenarios (of any type) in which the ego center travels at least
        minimum_threshold meters cumulatively. These are "non-stationary ego scenarios"
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type. Modified by function
    :param minimum_threshold: minimum distance in meters (inclusive, cumulative) the ego center has to travel in a given
        scenario for the scenario to be called a non-stationary ego scenario
    :return: Filtered scenario dictionary where the cumulative frame-to-frame displacement of the ego center in the
        scenario is >= the minimum threshold
    """
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = list(
            filter(lambda scenario: _is_non_stationary(scenario, minimum_threshold), scenario_dict[scenario_type])
        )
    return scenario_dict


class EdgeType(IntEnum):
    """
    Indices corresponding to relationships between two values adjacent in time.
    """

    RISING = 0
    FALLING = 1


def _check_for_speed_edge(
    scenario: NuPlanScenario, speed_threshold: float, speed_noise_tolerance: float, edge_type: EdgeType
) -> bool:
    """
    For a given scenario, determine whether there is a sub-scenario in which the ego's speed either
        rises above or falls below the speed_threshold.

    :param scenario: a NuPlan scenario
    :speed_threshold: what rear axle speed does the ego have to pass above (exclusive) to have "started moving?"
        likewise, what rear axle speed does the ego have to fall below (inclusive) to have "stopped moving?"
    :param speed_noise_tolerance: a value at or below which a speed change be ignored as noise.
    :param edge_type: are we filtering for speed RISING above the threshold or FALLING below the threshold?
    :return: a boolean, revealing whether a RISING/FALLING ego speed edge is present in the given scenario.
        or equal to the speed threshold and a subsequent frame in which the ego's speed is above the speed threshold.
        The second tells whether the scenario contains one frame in which the ego's speed is above the speed
        threshold and a subsequent frame in which the ego's speed is below or equal to the speed threshold.
    """
    if speed_noise_tolerance is None:
        speed_noise_tolerance = 0.1

    initial_ego_state = scenario.get_ego_state_at_iteration(0)
    current_speed, start_detector, stop_detector = (
        initial_ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude(),
    ) * 3

    edge_type_presence = [False, False]  # index 0 is presence of RISING, index 1 is presence of FALLING

    for next_ego_state in scenario.get_ego_future_trajectory(iteration=0, time_horizon=scenario.duration_s.time_s):
        next_speed = next_ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()

        if next_speed > start_detector:
            start_detector = next_speed
            if (
                start_detector > speed_threshold >= stop_detector
                and start_detector - stop_detector > speed_noise_tolerance
            ):
                edge_type_presence[EdgeType.RISING] = True
                stop_detector = start_detector

        if next_speed < stop_detector:
            stop_detector = next_speed
            if (
                start_detector > speed_threshold >= stop_detector
                and start_detector - stop_detector > speed_noise_tolerance
            ):
                edge_type_presence[EdgeType.FALLING] = True
                start_detector = stop_detector

    return edge_type_presence[edge_type]


def filter_ego_starts(
    scenario_dict: ScenarioDict, speed_threshold: float, speed_noise_tolerance: float
) -> ScenarioDict:
    """
    Filters a ScenarioDict, leaving only scenarios where the ego has started from a static position at some point

    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type. Modified by function
    :param speed_threshold: exclusive minimum velocity in meters per second that the ego rear axle must reach to be
        considered started
    :return: Filtered scenario dictionary where the ego reaches a speed greater than speed_threshold m/s from below
        at some point in all scenarios
    """
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = list(
            filter(
                lambda scenario: _check_for_speed_edge(
                    scenario, speed_threshold, speed_noise_tolerance, EdgeType.RISING
                ),
                scenario_dict[scenario_type],
            )
        )
    return scenario_dict


def filter_ego_stops(scenario_dict: ScenarioDict, speed_threshold: float, speed_noise_tolerance: float) -> ScenarioDict:
    """
    Filters a ScenarioDict, leaving only scenarios where the ego has stopped from a moving position at some point

    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type. Modified by function
    :param speed_threshold: inclusive maximum velocity in meters per second that the ego rear axle must reach to be
        considered stopped
    :return: Filtered scenario dictionary where the ego reaches a speed less than or equal to speed_threshold m/s
        from above at some point in all scenarios
    """
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = list(
            filter(
                lambda scenario: _check_for_speed_edge(
                    scenario, speed_threshold, speed_noise_tolerance, EdgeType.FALLING
                ),
                scenario_dict[scenario_type],
            )
        )
    return scenario_dict


def _ego_has_route(scenario: NuPlanScenario, map_radius: float) -> bool:
    """
    Determines the presence of an on-route lane segment in a VectorMap built from
    the given scenario within map_radius meters of the ego.
    :param scenario: A NuPlan scenario.
    :param map_radius: the radius of the VectorMap built around the ego's position
    to check for on-route lane segments.
    :return: True if there is at least one on-route lane segment in the VectorMap.
    """
    ego_state = scenario.initial_ego_state
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
    _, _, _, _, lane_seg_roadblock_ids = get_neighbor_vector_map(scenario.map_api, ego_coords, map_radius)
    map_lane_roadblock_ids = set(lane_seg_roadblock_ids.roadblock_ids)
    return len(map_lane_roadblock_ids.intersection(scenario.get_route_roadblock_ids())) > 0


def filter_ego_has_route(scenario_dict: ScenarioDict, map_radius: float) -> ScenarioDict:
    """
    Rid a scenario dictionary of the scenarios that don't have an on-route lane segment within map_radius meters of the ego.
    Uses a VectorMap to gather lane segments.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param map_radius: How far out from ego to check for on-route lane segments.
    :return: Filtered scenario dictionary.
    """
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = list(
            filter(lambda scenario: _ego_has_route(scenario, map_radius), scenario_dict[scenario_type])
        )
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


def get_scenarios_from_log_file(parameters: List[GetScenariosFromDbFileParams]) -> List[ScenarioDict]:
    """
    Gets all scenarios from a log file that match the provided parameters.
    :param parameters: The parameters to use for scenario extraction.
    :return: The extracted scenarios.
    """
    output_dict: ScenarioDict = {}
    for parameter in parameters:
        this_dict = get_scenarios_from_db_file(parameter)

        for key in this_dict:
            if key not in output_dict:
                output_dict[key] = this_dict[key]
            else:
                output_dict[key] += this_dict[key]

    return [output_dict]
