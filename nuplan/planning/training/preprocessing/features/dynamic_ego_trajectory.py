from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor


@dataclass
class DynamicEgoTrajectory(AbstractModelFeature):
    """
    Dataclass that holds trajectory signals produced from the model or from the dataset for supervision.
    Unlike Trajectory, this class represents the full dynamic state of ego.

    :param data: either a [num_batches, num_states, 11] or [num_states, 11] representing the trajectory
                 where the state is
                 (x [m], y [m], heading [rad],
                 longitudinal_velocity [m/s], lateral_velocity [m/s], angular_velocity [rad/s],
                 longitudinal_acceleration [m/s^2], lateral_acceleration [m/s^2], angular_acceleration [rad/s^2],
                 steering_angle [rad], steering_rate [rad/s])
    """

    data: FeatureDataType

    def __post_init__(self) -> None:
        """Implemented. See interface."""
        array_dims = self.num_dimensions
        state_size = self.data.shape[-1]

        if (array_dims != 2) and (array_dims != 3):
            raise RuntimeError(f"Invalid trajectory array. Expected 2 or 3 dims, got {array_dims}.")

        if state_size != self.state_size():
            raise RuntimeError(
                f"Invalid trajectory array. Expected {self.state_size()} variables per state, got {state_size}."
            )

    def to_device(self, device: torch.device) -> DynamicEgoTrajectory:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return DynamicEgoTrajectory(data=self.data.to(device=device))

    def to_feature_tensor(self) -> DynamicEgoTrajectory:
        """Inherited, see superclass."""
        return DynamicEgoTrajectory(data=to_tensor(self.data))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> DynamicEgoTrajectory:
        """Implemented. See interface."""
        return DynamicEgoTrajectory(data=data["data"])

    def unpack(self) -> List[DynamicEgoTrajectory]:
        """Implemented. See interface."""
        return [DynamicEgoTrajectory(data[None]) for data in self.data]

    @staticmethod
    def state_size() -> int:
        """
        Size of each dynamic state of the trajectory.
        """
        return 11

    @property
    def xy(self) -> FeatureDataType:
        """
        :return: tensor of positions [..., x, y]
        """
        return self.data[..., :2]

    @property
    def heading(self) -> FeatureDataType:
        """
        :return: tensor of positions [..., heading]
        """
        return self.data[..., 2]

    @property
    def velocity(self) -> FeatureDataType:
        """
        :return: tensor of velocities [..., longitudinal_velocity, lateral_velocity]
        """
        return self.data[..., 3:5]

    @property
    def angular_velocity(self) -> FeatureDataType:
        """
        :return: tensor of angular velocities [..., angular_velocity]
        """
        return self.data[..., 5]

    @property
    def acceleration(self) -> FeatureDataType:
        """
        :return: tensor of accelerations [..., longitudinal_acceleration, lateral_acceleration]
        """
        return self.data[..., 6:8]

    @property
    def angular_acceleration(self) -> FeatureDataType:
        """
        :return: tensor of angular accelerations [..., angular_acceleration]
        """
        return self.data[..., 8]

    @property
    def tire_steering_angle(self) -> FeatureDataType:
        """
        :return: tensor of tire steering angles [..., tire_steering_angle]
        """
        return self.data[..., 9]

    @property
    def tire_steering_rate(self) -> FeatureDataType:
        """
        :return: tensor of tire steering rates [..., tire_steering_rate]
        """
        return self.data[..., 10]

    @property
    def terminal_position(self) -> FeatureDataType:
        """
        :return: tensor of terminal position [..., x, y]
        """
        return self.data[..., -1, :2]

    @property
    def terminal_heading(self) -> FeatureDataType:
        """
        :return: tensor of terminal position [..., heading]
        """
        return self.data[..., -1, 2]

    @property
    def position_x(self) -> FeatureDataType:
        """
        Array of x positions of trajectory.
        """
        return self.data[..., 0]

    @property
    def numpy_position_x(self) -> FeatureDataType:
        """
        Numpy array of x positions of trajectory.
        """
        return np.asarray(self.data[..., 0])

    @property
    def position_y(self) -> FeatureDataType:
        """
        Array of y positions of trajectory.
        """
        return self.data[..., 1]

    @property
    def numpy_position_y(self) -> FeatureDataType:
        """
        Numpy array of y positions of trajectory.
        """
        return np.asarray(self.data[..., 1])

    @property
    def num_dimensions(self) -> int:
        """
        :return: dimensions of underlying data
        """
        return len(self.data.shape)

    @property
    def num_of_iterations(self) -> int:
        """
        :return: number of states in a trajectory
        """
        return int(self.data.shape[-2])

    @property
    def num_batches(self) -> Optional[int]:
        """
        :return: number of batches in the trajectory, None if trajectory does not have batch dimension
        """
        return None if self.num_dimensions <= 2 else self.data.shape[0]

    def state_at_index(self, index: int) -> FeatureDataType:
        """
        Query state at index along trajectory horizon
        :param index: along horizon
        :return: state corresponding to the index along trajectory horizon
        @raise in case index is not within valid range: 0 < index <= num_of_iterations
        """
        assert 0 <= index < self.num_of_iterations, f"Index is out of bounds! 0 <= {index} < {self.num_of_iterations}!"
        return self.data[..., index, :]

    def extract_number_of_last_states(self, number_of_states: int) -> DynamicEgoTrajectory:
        """
        Extract last number_of_states from a trajectory
        :param number_of_states: from last point
        :return: shorter trajectory containing number_of_states from end of trajectory
        @raise in case number_of_states is not within valid range: 0 < number_of_states <= length
        """
        assert number_of_states > 0, f"number_of_states has to be > 0, {number_of_states} > 0!"
        length = self.num_of_iterations
        assert (
            number_of_states <= length
        ), f"number_of_states has to be smaller than length, {number_of_states} <= {length}!"
        return self.extract_trajectory_between(length - number_of_states, length)

    def extract_trajectory_between(self, start_index: int, end_index: Optional[int]) -> DynamicEgoTrajectory:
        """
        Extract partial trajectory based on [start_index, end_index]
        :param start_index: starting index
        :param end_index: ending index
        :return: Trajectory
        @raise in case the desired ranges are not valid
        """
        if not end_index:
            end_index = self.num_of_iterations
        assert (
            0 <= start_index < self.num_of_iterations
        ), f"Start index is out of bounds! 0 <= {start_index} < {self.num_of_iterations}!"
        assert (
            0 <= end_index <= self.num_of_iterations
        ), f"Start index is out of bounds! 0 <= {end_index} <= {self.num_of_iterations}!"
        assert start_index < end_index, f"Start Index has to be smaller then end, {start_index} < {end_index}!"

        return DynamicEgoTrajectory(data=self.data[..., start_index:end_index, :])

    @classmethod
    def append_to_trajectory(cls, trajectory: DynamicEgoTrajectory, new_state: torch.Tensor) -> DynamicEgoTrajectory:
        """
        Extend trajectory with a new state, in this case we require that both trajectory and new_state has dimension
        of 3, that means that they both have batch dimension
        :param trajectory: to be extended
        :param new_state: state with which trajectory should be extended
        :return: extended trajectory
        """
        assert trajectory.num_dimensions == 3, f"Trajectory dimension {trajectory.num_dimensions} != 3!"
        assert len(new_state.shape) == 3, f"New state dimension {new_state.shape} != 3!"

        if new_state.shape[0] != trajectory.data.shape[0]:
            raise RuntimeError(f"Not compatible shapes {new_state.shape} != {trajectory.data.shape}!")

        if new_state.shape[-1] != trajectory.data.shape[-1]:
            raise RuntimeError(f"Not compatible shapes {new_state.shape} != {trajectory.data.shape}!")

        return DynamicEgoTrajectory(data=torch.cat((trajectory.data, new_state.clone()), dim=1))
