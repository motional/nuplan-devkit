from typing import List, cast

import torch
from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class ImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(self, weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'imitation_objective'
        self._weight = weight
        self._fn_xy = torch.nn.modules.loss.MSELoss(reduction='mean')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='mean')

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["trajectory"]

    def compute(self, predictions: FeaturesType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(Trajectory, predictions["trajectory"])
        targets_trajectory = cast(Trajectory, targets["trajectory"])

        return self._weight * (
            self._fn_xy(predicted_trajectory.xy, targets_trajectory.xy) +
            self._fn_heading(predicted_trajectory.heading, targets_trajectory.heading))
