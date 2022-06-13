import abc
from typing import Any, Dict, List

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class SceneConverter(abc.ABC):
    """
    Generic scene converter class.
    """

    @abc.abstractmethod
    def __call__(
        self, scenario: AbstractScenario, features: FeaturesType, targets: TargetsType, predictions: FeaturesType
    ) -> List[Dict[str, Any]]:
        """
        Convert selected information from scenario, features and targets into a list of scenes.
        The schema of the scene dictionary is defined by this function.
        :param scenario: scenario used for the data
        :param features: input features for the model
        :param targets: prediction targets
        :param predictions: prediction targets
        :return: list of scene dictionary for the scenario as well as relevant features and targets.
        """
        pass
