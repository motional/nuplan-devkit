from typing import List, Type, cast

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario, MockMapFactory
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class MockAbstractScenarioBuilder(AbstractScenarioBuilder):
    """Mock abstract scenario builder class used for testing."""

    @classmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """Inherited. See superclass."""
        return cast(Type[AbstractScenario], MockAbstractScenario)

    def get_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[AbstractScenario]:
        """Implemented. See interface."""
        return []

    def get_map_factory(self) -> MockMapFactory:
        """Implemented. See interface."""
        return MockMapFactory()
