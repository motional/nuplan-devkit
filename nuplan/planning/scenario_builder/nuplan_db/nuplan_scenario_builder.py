from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.database.nuplan_db.models import ScenarioTag
from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping, \
    extract_lidar_pcs_from_scenes, extract_scenes_from_log, filter_invalid_goals, flatten_scenarios
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.utils.multithreading.spliter import chunk_list
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool

logger = logging.getLogger(__name__)


def _create_scenarios(db: NuPlanDB,
                      scenario_tuples: List[Tuple[str, str]],
                      scenario_mapping: Optional[ScenarioMapping],
                      vehicle_parameters: VehicleParameters,
                      subsample_ratio: Optional[float] = None) -> List[AbstractScenario]:
    """
    Convert a list of scenario tokens to a list of AbstractScenario.
    :param db: DB object.
    :param scenario_tuples: List of tuples that define the scenario.
    :param scenario_mapping: Mapping of scenario types to extraction information.
    :param subsample_ratio: Ratio to subsample list of scenario lidarpcs.
    :param vehicle_parameters: Ego vehicle parameters
    :return: List of abstract scenarios corresponding to each input scenario tuple.
    """
    return [_create_scenario(db, scenario_tuple, scenario_mapping, vehicle_parameters, subsample_ratio)
            for scenario_tuple in scenario_tuples]


def _create_scenario(db: NuPlanDB,
                     scenario_tuple: Tuple[str, str],
                     scenario_mapping: Optional[ScenarioMapping],
                     vehicle_parameters: VehicleParameters,
                     subsample_ratio: Optional[float] = None) -> AbstractScenario:
    """
    Convert scenario tokens to AbstractScenario.
    :param db: DB object.
    :param scenario_tuple: Tuple that defines the scenario.
    :param scenario_mapping: Mapping of scenario types to extraction information.
    :param vehicle_parameters: Ego vehicle parameters
    :param subsample_ratio: Ratio to subsample list of scenario lidarpcs.
    :return: Instantiated AbstractScenario object.
    """
    scenario_type, lidar_token = scenario_tuple
    extraction_info = scenario_mapping.get_extraction_info(scenario_type) if scenario_mapping is not None else None
    scenario = NuPlanScenario(db=db, initial_lidar_token=lidar_token,
                              subsample_ratio=subsample_ratio,
                              scenario_extraction_info=extraction_info,
                              scenario_type=scenario_type,
                              ego_vehicle_parameters=vehicle_parameters)

    return scenario


def worker_map(worker: WorkerPool, fn: Callable[..., List[Any]], input_objects: List[Any]) -> List[Any]:
    """
    Maps a list of objects through a worker.
    :param worker: Worker pool to use for parallelization.
    :param fn: Function to use when mapping.
    :param input_objects: List of objects to map.
    :return: List of mapped objects.
    """
    if worker.number_of_threads == 0:
        return fn(input_objects)

    object_chunks = chunk_list(input_objects, worker.number_of_threads)
    scattered_objects = worker.map(Task(fn=fn), object_chunks)
    output_objects = [result for results in scattered_objects for result in results]

    return output_objects


class NuPlanScenarioBuilder(AbstractScenarioBuilder):
    def __init__(self,
                 version: str,
                 data_root: str,
                 scenario_mapping: ScenarioMapping = ScenarioMapping({}),
                 vehicle_parameters: VehicleParameters = get_pacifica_parameters()):
        """
        Build a scenario builder to retrieve scenarios from nuPlan DB.
        :param version: Version to load (e.g. "nuplan_v0.1.1").
        :param data_root: Path to the DB tables and blobs.
        :param scenario_mapping: Mapping of scenario types to extraction information.
        :param vehicle_parameters: Vehicle Parameters for this db.
        """
        self._db = NuPlanDB(version=version, data_root=data_root)
        self._scenario_mapping = scenario_mapping
        self._vehicle_parameters = vehicle_parameters

    def __reduce__(self) -> Tuple[Type[NuPlanScenarioBuilder], Tuple[Any, ...]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class
        """
        return self.__class__, (self._db.version, self._db.data_root, self._scenario_mapping, self._vehicle_parameters)

    @lru_cache(maxsize=8)
    def get_map_api(self, map_name: str) -> AbstractMap:
        """ Inherited. See superclass. """
        return NuPlanMapFactory(self._db.maps_db).build_map_from_name(map_name)

    def get_scenarios(self, scenario_filters: ScenarioFilters, worker: WorkerPool) -> List[AbstractScenario]:
        """ Implemented. See interface. """
        # Dictionary that holds a list of lidar tokens for each scenario type.
        scenario_dict: Dict[str, List[str]] = defaultdict(list)

        logger.info('Initial scenario filtering...')
        if scenario_filters.scenario_tokens is not None:  # Filter scenario by scenario token
            scenario_dict['unknown'] = scenario_filters.scenario_tokens
        elif scenario_filters.scenario_types is not None:  # Filter scenarios by desired scenario types
            assert self._scenario_mapping is not None, 'Cannot filter by scenario type, no scenario mapping found'
            available_types = [tag[0] for tag in self._db.session.query(ScenarioTag.type).distinct().all()]
            candidate_types = set(scenario_filters.scenario_types).intersection(available_types)
            tag_table = self._db.scenario_tag
            for scenario_type in candidate_types:
                scenario_dict[scenario_type] = [tag.lidar_pc_token for tag in tag_table.select_many(type=scenario_type)]
        elif scenario_filters.log_names:  # Filter logs based on log_names or tokens
            candidate_logs = [log for log in self._db.log if log.logfile in scenario_filters.log_names]
            if scenario_filters.max_scenarios_per_log:
                max_num = scenario_filters.max_scenarios_per_log
                scenes = [scene for log in candidate_logs for scene in extract_scenes_from_log(log)[:max_num]]
            else:
                scenes = [scene for log in candidate_logs for scene in extract_scenes_from_log(log)]
            scenario_dict['unknown'] = extract_lidar_pcs_from_scenes(scenes)
        else:  # Use all scenarios from each scene
            scenes = [scene for log in self._db.log for scene in extract_scenes_from_log(log)]
            scenario_dict['unknown'] = extract_lidar_pcs_from_scenes(scenes)
        logger.info('Initial scenario filtering...DONE')

        # Filter by the map version.
        if scenario_filters.map_name:
            map_name = scenario_filters.map_name
            logger.info(f"Selecting map version {map_name}...")
            scenario_dict = {scenario_type: [lidar_token for lidar_token in lidar_tokens
                                             if self._db.lidar_pc[lidar_token].log.map_version == map_name]
                             for scenario_type, lidar_tokens in scenario_dict.items()}
            logger.info(f"Selecting map version {map_name}...DONE")

        # Shuffle scenarios for each scenario type
        if scenario_filters.shuffle:
            logger.info('Shuffling scenarios for each scenario type...')
            for scenario_type in scenario_dict:
                random.shuffle(scenario_dict[scenario_type])
            logger.info('Shuffling scenarios for each scenario type...DONE')

        # Filter number of scenarios per scenario type
        if scenario_filters.limit_scenarios_per_type:
            logger.info('Limit scenario number by scenario type...')
            limit_scenarios = scenario_filters.limit_scenarios_per_type
            if isinstance(limit_scenarios, int):
                for scenario_type in scenario_dict:
                    scenario_dict[scenario_type] = scenario_dict[scenario_type][:limit_scenarios]
            elif isinstance(limit_scenarios, float):
                scenario_percentage = limit_scenarios
                for scenario_type in scenario_dict:
                    max_scenarios = math.ceil(len(scenario_dict[scenario_type]) * scenario_percentage)
                    scenario_dict[scenario_type] = scenario_dict[scenario_type][:max_scenarios]
            else:
                raise TypeError('Scenario filter "limit_total_scenarios" must be of type int or float')
            logger.info('Limit scenario number by scenario type...DONE')

        # Unravel dict to tuple for easier handling (shuffling, filtering etc.)
        scenario_tuples = [(name, token) for name, tokens in scenario_dict.items() for token in tokens]

        # Shuffle all scenarios
        if scenario_filters.shuffle:
            random.shuffle(scenario_tuples)

        # Expand to AbstractScenario objects
        logger.info("Converting scenarios to AbstractScenario...")
        scenario_mapping = self._scenario_mapping if not scenario_filters.flatten_scenarios else None
        if worker.number_of_threads > 1:
            scenario_chunks = chunk_list(scenario_tuples, worker.number_of_threads)
            scattered_objects = worker.map(Task(fn=_create_scenarios), self._db, scenario_chunks,
                                           scenario_mapping, self._vehicle_parameters,
                                           scenario_filters.subsample_ratio)
            scenario_objects = [result for results in scattered_objects for result in results]
        else:
            scenario_objects = worker.map(Task(fn=_create_scenario), self._db, scenario_tuples, scenario_mapping,
                                          self._vehicle_parameters, scenario_filters.subsample_ratio)
        logger.info("Converting scenarios to AbstractScenario...DONE")

        # Flatten scenarios
        if scenario_filters.flatten_scenarios:
            logger.info("Flattening multi-sample scenarios to single-sample scenarios...")
            scenario_objects = worker_map(worker, flatten_scenarios, scenario_objects)
            logger.info("Flattening multi-sample scenarios to single-sample scenarios...DONE")

        # Shuffle scenario objexts
        if scenario_filters.shuffle:
            random.shuffle(scenario_objects)

        # Remove scenarios that have invalid mission goals (e.g. nearby ego)
        if scenario_filters.remove_invalid_goals:
            logger.info("Removing scenarios with invalid goals...")
            scenario_objects = worker_map(worker, filter_invalid_goals, scenario_objects)
            logger.info("Removing scenarios with invalid goals...DONE")

        # Filter total number of scenarios
        if scenario_filters.limit_total_scenarios:
            logger.info(f'Limiting {len(scenario_objects)} total scenarios...')
            max_scenarios = scenario_filters.limit_total_scenarios
            if isinstance(max_scenarios, int):
                scenario_objects = scenario_objects[:max_scenarios]
            elif isinstance(max_scenarios, float):
                max_scenarios = math.ceil(len(scenario_objects) * max_scenarios)
                scenario_objects = scenario_objects[:max_scenarios]
            else:
                raise TypeError('Scenario filter "limit_total_scenarios" must be of type int or float')
            logger.info(f'Filtered {len(scenario_objects)} remaining scenarios...DONE')

        logger.info(f"Total number of extracted scenarios: {len(scenario_objects)}!")

        return scenario_objects
