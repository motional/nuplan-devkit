from abc import ABC, abstractmethod
from typing import List

import torch

from nuplan.planning.training.modeling.types import TargetsType


class AbstractTrainingMetric(ABC):
    """
    Abstract planning metric
    """

    @abstractmethod
    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        pass

    @abstractmethod
    def get_list_of_required_target_types(self) -> List[str]:
        """
        :return list of required targets for the computations
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Name of the metric
        """
        pass
