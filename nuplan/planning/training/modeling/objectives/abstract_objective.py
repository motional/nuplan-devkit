from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType


def aggregate_objectives(objectives: Dict[str, torch.Tensor], agg_mode: str) -> torch.Tensor:
    """
    Aggregates all computed objectives in a single scalar loss tensor used for backpropagation.

    :param objectives: dictionary of objective names and values
    :param agg_mode: how to aggregate multiple objectives. [mean, sum, max]
    :return: scalar loss tensor
    """
    if agg_mode == 'mean':
        return torch.stack(list(objectives.values())).mean()
    elif agg_mode == 'sum':
        return torch.stack(list(objectives.values())).sum()
    elif agg_mode == 'max':
        return torch.stack(list(objectives.values())).max()
    else:
        raise ValueError("agg_mode should be one of 'mean', 'sum', and 'max'.")


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
    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
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
