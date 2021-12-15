import abc
from typing import List

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class AbstractScenarioBuilder(abc.ABC):
    """ Interface for generic scenario builder. """

    @abc.abstractmethod
    def get_scenarios(self, scenario_filters: ScenarioFilters, worker: WorkerPool) -> List[AbstractScenario]:
        """
        Returns scenarios from the database.
        :return: A list of scenarios.
        """

        pass

    @abc.abstractmethod
    def get_map_api(self, map_name: str) -> AbstractMap:
        """
        Return a map database.
        :param map_name: Name of the map to be loaded.
        """

        pass
