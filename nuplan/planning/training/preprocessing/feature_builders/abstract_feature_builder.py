from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature

logger = logging.getLogger(__name__)


@dataclass
class FeatureBuilderMetaData:
    map_api: AbstractMap         # Abstract map api for accessing the maps.
    mission_goal: StateSE2       # Goal far into future (in generally more than 100m far beyond scenario length).
    expert_goal_state: StateSE2  # Expert state at the end of the scenario


class AbstractFeatureBuilder(ABC):
    """
    Abstract class that creates model input features from database samples.
    """

    @classmethod
    @abstractmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """
        :return A type of a feature that will be computed
        """
        pass

    @classmethod
    @abstractmethod
    def get_feature_unique_name(cls) -> str:
        """
        :return A unique string identifier of generated feature
        """
        pass

    @abstractmethod
    def get_features_from_simulation(self,
                                     ego_states: List[EgoState],
                                     observations: List[Observation],
                                     meta_data: FeatureBuilderMetaData) -> AbstractModelFeature:
        """
        Constructs model input features from simulation history.
        :param ego_states: Past ego state trajectory including the state at the current time step [t_-N, ..., t_-1, t_0]
        :param observations: Past observations including the observation at the current time step [t_-N, ..., t_-1, t_0]
        :param meta_data: Additional data require for building the feature
        :return: Constructed features
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
