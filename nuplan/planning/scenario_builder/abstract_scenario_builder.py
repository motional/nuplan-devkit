from __future__ import annotations

import abc
from enum import Enum
from typing import List, Type

from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class RepartitionStrategy(Enum):
    """Repartition strategy used when caching scenarios in a distributed setting."""

    REPARTITION_FILE_DISK = 1  # Loading scenarios from files, then redistribute to balance
    INLINE = 2  # Build all scenarios on each worker, then distribute evenly


class AbstractScenarioBuilder(abc.ABC):
    """Interface for generic scenario builder."""

    @classmethod
    @abc.abstractmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """Get the type of scenarios that this builder constructs."""
        pass

    @abc.abstractmethod
    def get_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[AbstractScenario]:
        """
        Retrieve filtered scenarios from the database.
        :param scenario_filter: Structure that contains scenario filtering instructions.
        :param worker: Worker pool for concurrent scenario processing.
        :return: A list of scenarios.
        """
        pass

    @abc.abstractmethod
    def get_map_factory(self) -> AbstractMapFactory:
        """
        Get a map factory instance.
        """
        pass

    @property
    def repartition_strategy(self) -> RepartitionStrategy:
        """
        Gets the repartition strategy used for caching in a distributed setting.
        """
        pass
