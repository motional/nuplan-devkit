from typing import List

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractMap
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class MockAbstractScenarioBuilder(AbstractScenarioBuilder):

    def get_scenarios(self, scenario_filters: ScenarioFilters, worker: WorkerPool) -> List[AbstractScenario]:
        """ Implemented. See interface. """
        return []

    def get_map_api(self, map_name: str) -> MockAbstractMap:
        """ Implemented. See interface. """

        return MockAbstractMap()
