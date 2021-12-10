from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


def aggregate_objectives(objectives: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Aggregates all computed objectives in a single scalar loss tensor used for backpropagation.

    :param objectives: dictionary of objective names and values
    :return: scalar loss tensor
    """
    return torch.stack(list(objectives.values())).mean()


class AbstractObjective(ABC):
    """
    Abstract learning objective class.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Name of the objective
        """
        pass

    @abstractmethod
    def compute(self, predictions: FeaturesType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        pass

    @abstractmethod
    def get_list_of_required_target_types(self) -> List[str]:
        """
        :return list of required targets for the computations
        """
        pass
