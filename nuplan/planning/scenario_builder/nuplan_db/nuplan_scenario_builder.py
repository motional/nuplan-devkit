from __future__ import annotations

import logging
from functools import partial
from typing import Any, List, Optional, Tuple, Type, Union, cast

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.database.nuplan_db.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import (
    FilterWrapper,
    ScenarioDict,
    create_all_scenarios,
    create_scenarios_by_tokens,
    create_scenarios_by_types,
    filter_by_map_names,
    filter_invalid_goals,
    filter_num_scenarios_per_type,
    filter_total_num_scenarios,
    scenario_dict_to_list,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)


class NuPlanScenarioBuilder(AbstractScenarioBuilder):
    """Builder class for constructing nuPlan scenarios for training and simulation."""

    def __init__(
        self,
        data_root: str,
        map_root: str,
        db_files: Optional[Union[List[str], str]],
        map_version: str,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        scenario_mapping: Optional[ScenarioMapping] = None,
        vehicle_parameters: Optional[VehicleParameters] = None,
        ground_truth_predictions: Optional[TrajectorySampling] = None,
    ):
        """
        Initialize scenario builder that filters and retrieves scenarios from the nuPlan dataset.
        :param data_root: Local data root for loading (or storing downloaded) the log databases.
                          If `db_files` is not None, all downloaded databases will be stored to this data root.
        :param map_root: Local map root for loading (or storing downloaded) the map database.
        :param db_files: Path to load the log database(s) from.
                         It can be a local/remote path to a single database, list of databases or dir of databases.
                         If None, all database filenames found under `data_root` will be used.
        :param map_version: Version of map database to load. The map database is passed to each loaded log database.
        :param max_workers: Maximum number of workers to use when loading the databases concurrently.
                            Only used when the number of databases to load is larger than this parameter.
        :param verbose: Whether to print progress and details during the database loading and scenario building.
        :param scenario_mapping: Mapping of scenario types to extraction information.
        :param vehicle_parameters: Vehicle parameters for this db.
        """
        self._data_root = data_root
        self._map_root = map_root
        self._db_files = db_files
        self._map_version = map_version
        self._max_workers = max_workers
        self._verbose = verbose
        self._ground_truth_predictions = ground_truth_predictions
        self._scenario_mapping = scenario_mapping if scenario_mapping is not None else ScenarioMapping({})
        self._vehicle_parameters = vehicle_parameters if vehicle_parameters is not None else get_pacifica_parameters()

        self._db = NuPlanDBWrapper(
            data_root=data_root,
            map_root=map_root,
            db_files=db_files,
            map_version=map_version,
            max_workers=max_workers,
            verbose=verbose,
        )

    def __reduce__(self) -> Tuple[Type[NuPlanScenarioBuilder], Tuple[Any, ...]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class
        """
        return self.__class__, (
            self._data_root,
            self._map_root,
            self._db_files,
            self._map_version,
            self._max_workers,
            self._verbose,
            self._scenario_mapping,
            self._vehicle_parameters,
            self._ground_truth_predictions,
        )

    @classmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """Inherited. See superclass."""
        return cast(Type[AbstractScenario], NuPlanScenario)

    def get_map_factory(self) -> NuPlanMapFactory:
        """Inherited. See superclass."""
        return NuPlanMapFactory(self._db.maps_db)

    def _create_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> ScenarioDict:
        """
        Creates a scenario dictionary with scenario type as key and list of scenarios for each type.
        :param scenario_filter: Structure that contains scenario filtering instructions.
        :param worker: Worker pool for concurrent scenario processing.
        :return: Constructed scenario dictionary.
        """
        if scenario_filter.scenario_tokens is not None:  # Filter scenarios by desired scenario tokens
            scenario_dict = create_scenarios_by_tokens(
                scenario_filter.scenario_tokens,
                self._db,
                scenario_filter.log_names,
                scenario_filter.expand_scenarios,
                self._vehicle_parameters,
                self._ground_truth_predictions,
            )
        elif scenario_filter.scenario_types is not None:  # Filter scenarios by desired scenario types
            scenario_dict = create_scenarios_by_types(
                scenario_filter.scenario_types,
                self._db,
                scenario_filter.log_names,
                scenario_filter.expand_scenarios,
                self._scenario_mapping,
                self._vehicle_parameters,
                self._ground_truth_predictions,
            )
        else:  # Use all scenarios from each scene
            scenario_dict = create_all_scenarios(
                self._db,
                scenario_filter.log_names,
                scenario_filter.expand_scenarios,
                self._vehicle_parameters,
                worker,
                self._ground_truth_predictions,
            )

        return scenario_dict

    def _create_filter_wrappers(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[FilterWrapper]:
        """
        Creates a series of filter wrappers that will be applied sequentially to construct the list of scenarios.
        :param scenario_filter: Structure that contains scenario filtering instructions.
        :param worker: Worker pool for concurrent scenario processing.
        :return: Series of filter wrappers.
        """
        filters = [
            FilterWrapper(
                fn=partial(filter_by_map_names, map_names=scenario_filter.map_names, db=self._db),
                enable=(scenario_filter.map_names is not None),
                name='map_names',
            ),
            FilterWrapper(
                fn=partial(
                    filter_num_scenarios_per_type,
                    num_scenarios_per_type=scenario_filter.num_scenarios_per_type,
                    randomize=scenario_filter.shuffle,
                ),
                enable=(scenario_filter.num_scenarios_per_type is not None),
                name='num_scenarios_per_type',
            ),
            FilterWrapper(
                fn=partial(
                    filter_total_num_scenarios,
                    limit_total_scenarios=scenario_filter.limit_total_scenarios,
                    randomize=scenario_filter.shuffle,
                ),
                enable=(scenario_filter.limit_total_scenarios is not None),
                name='limit_total_scenarios',
            ),
            FilterWrapper(
                fn=partial(filter_invalid_goals, worker=worker),
                enable=scenario_filter.remove_invalid_goals,
                name='remove_invalid_goals',
            ),
        ]

        return filters

    def get_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[AbstractScenario]:
        """Implemented. See interface."""
        # Create scenario dictionary and series of filters to apply
        scenario_dict = self._create_scenarios(scenario_filter, worker)
        filter_wrappers = self._create_filter_wrappers(scenario_filter, worker)

        # Apply filtering strategy sequentially to the scenario dictionary
        for filter_wrapper in filter_wrappers:
            scenario_dict = filter_wrapper.run(scenario_dict)

        return scenario_dict_to_list(scenario_dict, shuffle=scenario_filter.shuffle)  # type: ignore
