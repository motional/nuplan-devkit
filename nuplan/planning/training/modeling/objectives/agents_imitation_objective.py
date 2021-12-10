from typing import List, cast

import torch
from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories


class AgentsImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(self, name: str = 'agent_imitation_objective', weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = name
        self._weight = weight
        self._fn = torch.nn.modules.loss.MSELoss(reduction='mean')

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """ Implemented. See interface. """
        return ["agents"]

    def compute(self, predictions: FeaturesType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(AgentsTrajectories, predictions["agents"])
        targets_trajectory = cast(AgentsTrajectories, targets["agents"])
        batch_size = predicted_trajectory.batch_size

        loss = 0.0
        for sample_idx in range(batch_size):
            loss += self._fn(predicted_trajectory.poses[sample_idx], targets_trajectory.poses[sample_idx])

        return self._weight * loss / batch_size
