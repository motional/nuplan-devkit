from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, \
    FeatureDataType, to_tensor
from pyquaternion import Quaternion


@dataclass
class AgentsFeature(AbstractModelFeature):
    """
    Model input feature representing the present and past states of the ego and agents.

    The structure inludes:
        ego: List[<np.ndarray: num_frames, 3>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The last dimension is the ego pose (x, y, heading) at time t.
            Example dimensions: 8 (batch_size) x 5 (1 present + 4 past frames) x 3
        agents: List[<np.ndarray: num_frames, num_agents, 8>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The num_agents is padded to fit the largest number of agents across all frames.
            The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate)
             and size (length, width) at time t.

    The present/past frames dimension is populated in increasing chronological order, i.e. (t_-N, ..., t_-1, t_0)
    where N is the number of frames in the feature

    In both cases, the outer List represent number of batches. This is a special feature where each batch entry
    can have different size. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """
    ego: List[FeatureDataType]
    agents: List[FeatureDataType]

    def __post_init__(self) -> None:
        if len(self.ego) != len(self.agents):
            raise AssertionError(
                f"Not consistent length of batches! {len(self.ego)} != {len(self.agents)}")

        if len(self.ego) == 0:
            raise AssertionError("Batch size has to be > 0!")

        if self.ego[0].ndim != 2:
            raise AssertionError("Ego feature samples does not conform to feature dimensions! "
                                 f"Got ndim: {self.ego[0].ndim} , expected 2 [num_frames, 3]")

        if self.agents[0].ndim != 3:
            raise AssertionError("Agent feature samples does not conform to feature dimensions! "
                                 f"Got ndim: {self.agents[0].ndim} , "
                                 f"expected 3 [num_frames, num_agents, 8]")

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        return len(self.ego)

    @classmethod
    def collate(cls, batch: List[AgentsFeature]) -> AgentsFeature:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return AgentsFeature(ego=[item.ego[0] for item in batch], agents=[item.agents[0] for item in batch])

    def to_feature_tensor(self) -> AgentsFeature:
        """ Implemented. See interface. """
        return AgentsFeature(ego=[to_tensor(ego) for ego in self.ego],
                             agents=[to_tensor(agents) for agents in self.agents])

    def to_device(self, device: torch.device) -> AgentsFeature:
        """ Implemented. See interface. """
        return AgentsFeature(ego=[ego.to(device=device) for ego in self.ego],
                             agents=[agents.to(device=device) for agents in self.agents])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AgentsFeature:
        """ Implemented. See interface. """
        return AgentsFeature(ego=data["ego"], agents=data["agents"])

    def num_agents_in_sample(self, sample_idx: int) -> int:
        """
        Returns the number of agents at a given batch
        :param sample_idx: the batch index of interest
        :return: number of agents in the given batch
        """
        return self.agents[sample_idx].shape[1]  # type: ignore

    @staticmethod
    def ego_state_dim() -> int:
        """
        :return: ego state dimension
        """
        return 3

    @staticmethod
    def agents_states_dim() -> int:
        """
        :return: agent state dimension
        """
        return 8

    @property
    def num_frames(self) -> int:
        """
        :return: number of frames.
        """
        return int(self.agents[0].shape[0])

    @property
    def ego_feature_dim(self) -> int:
        """
        :return: ego feature dimension. Note, the plus one is to account for the present frame
        """
        return AgentsFeature.ego_state_dim() * self.num_frames

    @property
    def agents_features_dim(self) -> int:
        """
        :return: ego feature dimension. Note, the plus one is to account for the present frame
        """
        return AgentsFeature.agents_states_dim() * self.num_frames

    def has_agents(self, batch_idx: int) -> bool:
        """
        Check whether agents exist in the feature.
        :param batch_idx: the batch index of interest
        :return: whether agents exist in the feature
        """
        return self.num_agents_in_sample(batch_idx) > 0

    def get_flatten_agents_features_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Flatten agents' features by stacking the agents' states along the num_frame dimension
        <np.ndarray: num_frames, num_agents, 8>] -> <np.ndarray: num_agents, num_frames x 8>]

        :param sample_idx: the sample index of interest
        :return: <FeatureDataType: num_agents, num_frames x 8>] agent feature
        """
        data = self.agents[sample_idx]
        axes = (1, 0) if isinstance(data, torch.Tensor) else (1, 0, 2)
        return data.transpose(*axes).reshape(data.shape[1], -1)

    def get_ego_agents_center_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return ego center in the given batch
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 2>. (x, y) positions of the ego's center at sample time step
        """
        return self.ego[sample_idx][0][:2]

    def get_agents_centers_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents' centers in the given sample
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: num_agents, 2>. (x, y) positions of the agents' the centers of all agents at the
         sample time step
        """
        return self.agents[sample_idx][0][:, :2]

    def rotate(self, quaternion: Quaternion) -> AgentsFeature:
        raise NotImplementedError

    def translate(self, translation_value: FeatureDataType) -> AgentsFeature:
        raise NotImplementedError

    def scale(self, scale_value: FeatureDataType) -> AgentsFeature:
        raise NotImplementedError

    def xflip(self) -> AgentsFeature:
        raise NotImplementedError

    def yflip(self) -> AgentsFeature:
        raise NotImplementedError
