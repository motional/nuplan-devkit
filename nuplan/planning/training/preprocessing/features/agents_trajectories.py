from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, \
    FeatureDataType, to_tensor


@dataclass
class AgentsTrajectories(AbstractModelFeature):
    """
    Model input feature representing the present and past states of the ego and agents.

    The structure inludes:
        agents: List[<np.ndarray: num_frames, num_agents, 6>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The num_agents is padded to fit the largest number of agents across all frames.
            The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate) at time t.

    The present/future frames dimension is populated in ascending chronological order, i.e. (t_1, t_2, ..., t_n)

    The outer List represent number of batches. This is a special feature where each batch entry
    can have different size. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """

    data: List[FeatureDataType]

    def __post_init__(self) -> None:
        if len(self.data) == 0:
            raise AssertionError("Batch size has to be > 0!")

    @property
    def batch_size(self) -> int:
        """
        :return: batch size
        """
        return len(self.data)

    @staticmethod
    def states_dim() -> int:
        """
        :return: agent state dimension
        """
        return 6

    @property
    def num_frames(self) -> int:
        """
        :return: number of future frames. Note: this excludes the present frame
        """
        return int(self.data[0].shape[0])

    @property
    def features_dim(self) -> int:
        """
        :return: ego feature dimension
        """
        return self.num_frames * AgentsTrajectories.states_dim()

    @classmethod
    def collate(cls, batch: List[AgentsTrajectories]) -> AgentsTrajectories:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return AgentsTrajectories(data=[item.data[0] for item in batch])

    def to_feature_tensor(self) -> AgentsTrajectories:
        """ Implemented. See interface. """
        return AgentsTrajectories(data=[to_tensor(data) for data in self.data])

    def to_device(self, device: torch.device) -> AgentsTrajectories:
        """ Implemented. See interface. """
        return AgentsTrajectories(data=[data.to(device=device) for data in self.data])

    @ classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AgentsTrajectories:
        """ Implemented. See interface. """
        return AgentsTrajectories(data=data["data"])

    def num_agents_in_sample(self, sample_idx: int) -> int:
        """
        Returns the number of agents at a given batch
        :param sample_idx: the batch index of interest
        :return: number of agents in the given batch
        """
        return self.data[sample_idx].shape[1]  # type: ignore

    def has_agents(self, batch_idx: int) -> bool:
        """
        Check whether agents exist in the feature.
        :param batch_idx: the batch index of interest
        :return: whether agents exist in the feature
        """
        return self.num_agents_in_sample(batch_idx) > 0

    @property
    def xy(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: num_frames, num_agents, 3>] x, y of all agent across all frames
        """
        return [sample[..., :2] for sample in self.data]

    @property
    def heading(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: num_frames, num_agents, 3>] yaw of all agent across all frames
        """
        return [sample[..., 3] for sample in self.data]

    @property
    def poses(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: num_frames, num_agents, 3>] x, y, yaw of all agent across all frames
        """
        return [sample[..., :3] for sample in self.data]

    @property
    def velocities(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: num_frames, num_agents, 3>] x, y, yaw of all agent across all frames
        """
        return [[agent[..., 3:6] for agent in sample] for sample in self.data]

    @property
    def terminal_xy(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: terminal_frame, num_agents, 3>] x, y of all agent across all frames
        """
        return [sample[-1, :, :2] for sample in self.data]

    @property
    def terminal_heading(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: terminal_frame, num_agents, 3>] x, y of all agent across all frames
        """
        return [sample[-1, :, 3] for sample in self.data]
