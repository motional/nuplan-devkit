from typing import List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.dynamic_ego_trajectory import DynamicEgoTrajectory


class DynamicImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories,
    including the dynamic components of the state.
    """

    def __init__(
        self,
        overall_weight: float = 1.0,
        x_position_weight: float = 1.0,
        y_position_weight: float = 1.0,
        heading_weight: float = 1.0,
        longitudinal_velocity_weight: float = 1.0,
        lateral_velocity_weight: float = 1.0,
        angular_velocity_weight: float = 1.0,
        longitudinal_acceleration_weight: float = 1.0,
        lateral_acceleration_weight: float = 1.0,
        angular_acceleration_weight: float = 1.0,
        steering_angle_weight: float = 1.0,
        steering_rate_weight: float = 1.0,
        loss_function: torch.nn.Module = torch.nn.modules.loss.L1Loss(reduction='mean'),
    ) -> None:
        """
        Initializes the class

        :param overall_weight: weight contribution to the overall loss
        :param x_position_weight: weight on loss in x-position
        :param y_position_weight: weight on loss in y-position
        :param heading_position_weight: weight on loss in heading
        :param longitudinal_velocity_weight: weight on loss in longitudinal velocity
        :param lateral_velocity_weight: weight on loss in lateral velocity
        :param angular_velocity_weight: weight on loss in angular velocity
        :param longitudinal_acceleration_weight: weight on loss in longitudinal acceleration
        :param lateral_acceleration_weight: weight on loss in lateral acceleration
        :param angular_acceleration_weight: weight on loss in angular acceleration
        :param steering_angle_weight: weight on loss in steering angle
        :param steering_rate_weight: weight on loss in steering rate
        """
        self._name = 'dynamic_imitation_objective'
        self._overall_weight = overall_weight
        self._x_position_weight = x_position_weight
        self._y_position_weight = y_position_weight
        self._heading_weight = heading_weight
        self._longitudinal_velocity_weight = longitudinal_velocity_weight
        self._lateral_velocity_weight = lateral_velocity_weight
        self._angular_velocity_weight = angular_velocity_weight
        self._longitudinal_acceleration_weight = longitudinal_acceleration_weight
        self._lateral_acceleration_weight = lateral_acceleration_weight
        self._angular_acceleration_weight = angular_acceleration_weight
        self._steering_angle_weight = steering_angle_weight
        self._steering_rate_weight = steering_rate_weight
        self._loss_function = loss_function

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["dynamic_ego_trajectory"]

    def compute(self, predictions: FeaturesType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on the fixed weights.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(DynamicEgoTrajectory, predictions["dynamic_ego_trajectory"])
        targets_trajectory = cast(DynamicEgoTrajectory, targets["dynamic_ego_trajectory"])

        loss = self._x_position_weight * self._loss_function(
            predicted_trajectory.xy[..., 0], targets_trajectory.xy[..., 0]
        )
        loss += self._y_position_weight * self._loss_function(
            predicted_trajectory.xy[..., 1], targets_trajectory.xy[..., 1]
        )
        loss += self._heading_weight * self._loss_function(
            torch.sin(predicted_trajectory.heading), torch.sin(targets_trajectory.heading)
        )
        loss += self._heading_weight * self._loss_function(
            torch.cos(predicted_trajectory.heading), torch.cos(targets_trajectory.heading)
        )
        loss += self._longitudinal_velocity_weight * self._loss_function(
            predicted_trajectory.velocity[..., 0], targets_trajectory.velocity[..., 0]
        )
        loss += self._lateral_velocity_weight * self._loss_function(
            predicted_trajectory.velocity[..., 1], targets_trajectory.velocity[..., 1]
        )
        loss += self._angular_velocity_weight * self._loss_function(
            predicted_trajectory.angular_velocity, targets_trajectory.angular_velocity
        )
        loss += self._longitudinal_acceleration_weight * self._loss_function(
            predicted_trajectory.acceleration[..., 0], targets_trajectory.acceleration[..., 0]
        )
        loss += self._lateral_acceleration_weight * self._loss_function(
            predicted_trajectory.acceleration[..., 1], targets_trajectory.acceleration[..., 1]
        )
        loss += self._angular_acceleration_weight * self._loss_function(
            predicted_trajectory.angular_acceleration, targets_trajectory.angular_acceleration
        )
        loss += self._steering_angle_weight * self._loss_function(
            torch.sin(predicted_trajectory.tire_steering_angle), torch.sin(targets_trajectory.tire_steering_angle)
        )
        loss += self._steering_angle_weight * self._loss_function(
            torch.cos(predicted_trajectory.tire_steering_angle), torch.cos(targets_trajectory.tire_steering_angle)
        )
        loss += self._steering_rate_weight * self._loss_function(
            predicted_trajectory.tire_steering_rate, targets_trajectory.tire_steering_rate
        )

        return self._overall_weight * loss
