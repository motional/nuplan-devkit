from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature


class AbstractTargetBuilder(ABC):
    """
    Abstract class that creates model output targets from database samples.
    """

    @classmethod
    @abstractmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """
        :return type of feature which will be generated
        """
        pass

    @classmethod
    @abstractmethod
    def get_feature_unique_name(cls) -> str:
        """
        :return a unique string identifier of generated feature
        """
        pass

    @abstractmethod
    def get_targets(self, scenario: AbstractScenario) -> AbstractModelFeature:
        """
        Constructs model output targets from database scenario.

        :param scenario: generic scenario
        :return: constructed targets
        """
        pass
