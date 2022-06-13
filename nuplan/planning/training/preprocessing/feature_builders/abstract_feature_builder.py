from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature


class AbstractFeatureBuilder(ABC):
    """
    Abstract class that creates model input features from database samples.
    """

    @classmethod
    @abstractmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        pass

    @classmethod
    @abstractmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        pass

    @abstractmethod
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AbstractModelFeature:
        """
        Constructs model input features from simulation history.
        :param current_input: Iteration specific inputs for building the feature.
        :param initialization: Additional data require for building the feature.
        :return: Constructed features.
        """
        pass

    @abstractmethod
    def get_features_from_scenario(self, scenario: AbstractScenario) -> AbstractModelFeature:
        """
        Constructs model input features from a database samples.
        :param scenario: Generic scenario
        :return: Constructed features
        """
        pass
