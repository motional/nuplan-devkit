from __future__ import annotations

from typing import List, Type, cast

from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder, RepartitionStrategy
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario, MockMapFactory
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class MockAbstractScenarioBuilder(AbstractScenarioBuilder):
    """Mock abstract scenario builder class used for testing."""

    def __init__(self, num_scenarios: int = 0):
        """
        The init method
        :param num_scenarios: The number of scenarios to return from get_scenarios()
        """
        self.num_scenarios = num_scenarios

    @classmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """Inherited. See superclass."""
        return cast(Type[AbstractScenario], MockAbstractScenario)

    def get_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[AbstractScenario]:
        """Implemented. See interface."""
        return [MockAbstractScenario() for _ in range(self.num_scenarios)]

    def get_map_factory(self) -> AbstractMapFactory:
        """Implemented. See interface."""
        return MockMapFactory()

    @property
    def repartition_strategy(self) -> RepartitionStrategy:
        """Implemented. See interface."""
        return RepartitionStrategy.INLINE
