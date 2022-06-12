from typing import List

import torch

from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories


class AgentsAverageDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of all agents' trajectory.
    """

    def __init__(self, name: str = 'agents_avg_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: AgentsTrajectories = predictions["agents_trajectory"]
        target_agents: AgentsTrajectories = targets["agents_trajectory"]
        batch_size = predicted_agents.batch_size

        error = torch.mean(
            torch.tensor(
                [
                    torch.norm(predicted_agents.xy[sample_idx] - target_agents.xy[sample_idx], dim=-1).mean()
                    for sample_idx in range(batch_size)
                ]
            )
        )

        return error


class AgentsFinalDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from the final pose of all agents trajectory.
    """

    def __init__(self, name: str = 'agents_final_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: AgentsTrajectories = predictions["agents_trajectory"]
        target_agents: AgentsTrajectories = targets["agents_trajectory"]
        batch_size = predicted_agents.batch_size

        error = torch.mean(
            torch.tensor(
                [
                    torch.norm(
                        predicted_agents.terminal_xy[sample_idx] - target_agents.terminal_xy[sample_idx], dim=-1
                    ).mean()
                    for sample_idx in range(batch_size)
                ]
            )
        )
        return error


class AgentsAverageHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error averaged from all poses of all agents trajectory.
    """

    def __init__(self, name: str = 'agents_avg_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: AgentsTrajectories = predictions["agents_trajectory"]
        target_agents: AgentsTrajectories = targets["agents_trajectory"]
        batch_size = predicted_agents.batch_size

        errors = []
        for sample_idx in range(batch_size):
            error = torch.abs(predicted_agents.heading[sample_idx] - target_agents.heading[sample_idx])
            error_wrapped = torch.atan2(torch.sin(error), torch.cos(error)).mean()
            errors.append(error_wrapped)
        return torch.mean(torch.tensor(errors))


class AgentsFinalHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error from the final pose of all agents agents.
    """

    def __init__(self, name: str = 'agents_final_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: AgentsTrajectories = predictions["agents_trajectory"]
        target_agents: AgentsTrajectories = targets["agents_trajectory"]
        batch_size = predicted_agents.batch_size

        errors = []
        for sample_idx in range(batch_size):
            error = torch.abs(
                predicted_agents.terminal_heading[sample_idx] - target_agents.terminal_heading[sample_idx]
            )
            error_wrapped = torch.atan2(torch.sin(error), torch.cos(error)).mean()
            errors.append(error_wrapped)

        return torch.mean(torch.tensor(errors))
