from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)


@dataclass
class DynamicEgoFeature(AbstractModelFeature):
    """
    Model feature representing dynamic ego state, including velocity, acceleration and steering angle.

    The structure includes:
        ego: List[<np.ndarray: num_frames, 11>]
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The last dimension is the dynamic state for rear axle at time t
                (x, y, heading, longitudinal_velocity, lateral_velocity, angular_velocity,
                longitudinal_acceleration, lateral_acceleration, angular_acceleration,
                steering_angle, steering_rate).
    """

    ego: List[FeatureDataType]

    def __post_init__(self) -> None:
        """Implemented. See interface."""
        if len(self.ego) == 0:
            raise AssertionError("Batch size has to be > 0!")

        for batch_sample in self.ego:
            if batch_sample.ndim != 2:
                print(batch_sample)
                raise AssertionError(
                    "Ego feature samples does not conform to feature dimensions! "
                    f"Got ndim: {self.ego[0].ndim} , expected 2 [num_frames, {self.ego_state_dim}]"
                )
            if batch_sample.shape[1] != self.ego_state_dim():
                raise AssertionError(
                    "Ego feature samples does not conform to state size! "
                    f"Got ndim: {self.ego[0].shape[1]} , expected {self.ego_state_dim}"
                )

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        return len(self.ego)

    @classmethod
    def collate(cls, batch: List[DynamicEgoFeature]) -> DynamicEgoFeature:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return DynamicEgoFeature(ego=[item.ego[0] for item in batch])

    def to_feature_tensor(self) -> DynamicEgoFeature:
        """Implemented. See interface."""
        return DynamicEgoFeature(ego=[to_tensor(ego) for ego in self.ego])

    def to_device(self, device: torch.device) -> DynamicEgoFeature:
        """Implemented. See interface."""
        return DynamicEgoFeature(ego=[ego.to(device=device) for ego in self.ego])

    @property
    def device(self) -> torch.device:
        """Current device of feature data."""
        return self.ego[0].device

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> DynamicEgoFeature:
        """Implemented. See interface."""
        return DynamicEgoFeature(ego=data["ego"])

    def unpack(self) -> List[DynamicEgoFeature]:
        """Implemented. See interface."""
        return [DynamicEgoFeature([ego]) for ego in self.ego]

    @staticmethod
    def ego_state_dim() -> int:
        """
        :return: ego state dimension
        """
        return 11

    def get_present_ego_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present ego in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 11>. ego at sample index
        """
        return self.ego[sample_idx][-1]

    def get_present_ego_pose_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present ego pose in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 3>. rear axle ego pose (x, y, heading) at sample index
        """
        return self.ego[sample_idx][-1][:3]

    def get_present_ego_velocity_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present ego velocity in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 3>. rear axle ego velocity (longitudinal, lateral, angular) at sample index
        """
        return self.ego[sample_idx][-1][3:6]

    def get_present_ego_acceleration_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present ego acceleration in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 3>. rear axle ego acceleration (longitudinal, lateral, angular) at sample index
        """
        return self.ego[sample_idx][-1][6:9]

    def get_present_ego_tire_steering_angle_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present ego tire steering angle in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 1>. tire steering angle at sample index
        """
        return self.ego[sample_idx][-1][9:10]

    def get_present_ego_tire_steering_rate_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present ego tire steering rate in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 1>. tire steering rate at sample index
        """
        return self.ego[sample_idx][-1][10:11]
