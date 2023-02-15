from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)


@dataclass
class GenericAgents(AbstractModelFeature):
    """
    Model input feature representing the present and past states of the ego and agents.

    The structure includes:
        ego: List[<np.ndarray: num_frames, 7>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The last dimension is the ego pose (x, y, heading) velocities (vx, vy) accelerations (ax, ay) at time t.
            Example dimensions: 8 (batch_size) x 5 (1 present + 4 past frames) x 7
        agents: Dict[str, List[<np.ndarray: num_frames, num_agents, 8>]].
            Agent features indexed by agent feature type.
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
    agents: Dict[str, List[FeatureDataType]]

    def __post_init__(self) -> None:
        """Sanitize attributes of dataclass."""
        if not all([len(self.ego) == len(agent) for agent in self.agents.values()]):
            raise AssertionError("Batch size inconsistent across features!")

        if len(self.ego) == 0:
            raise AssertionError("Batch size has to be > 0!")

        if self.ego[0].ndim != 2:
            raise AssertionError(
                "Ego feature samples does not conform to feature dimensions! "
                f"Got ndim: {self.ego[0].ndim} , expected 2 [num_frames, 7]"
            )

        if 'EGO' in self.agents.keys():
            raise AssertionError("EGO not a valid agents feature type!")
        for feature_name in self.agents.keys():
            if feature_name not in TrackedObjectType._member_names_:
                raise ValueError(f"Object representation for layer: {feature_name} is unavailable!")

        for agent in self.agents.values():
            if agent[0].ndim != 3:
                raise AssertionError(
                    "Agent feature samples does not conform to feature dimensions! "
                    f"Got ndim: {agent[0].ndim} , "
                    f"expected 3 [num_frames, num_agents, 8]"
                )

        for sample_idx in range(len(self.ego)):
            if int(self.ego[sample_idx].shape[0]) != self.num_frames or not all(
                [int(agent[sample_idx].shape[0]) == self.num_frames for agent in self.agents.values()]
            ):
                raise AssertionError("Agent feature samples have different number of frames!")

    def _validate_ego_query(self, sample_idx: int) -> None:
        """
        Validate ego sample query is valid.
        :param sample_idx: the batch index of interest.
        :raise
            ValueError if sample_idx invalid.
            RuntimeError if feature at given sample index is empty.
        """
        if self.batch_size < sample_idx:
            raise ValueError(f"Requsted sample index {sample_idx} larger than batch size {self.batch_size}!")
        if self.ego[sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")

    def _validate_agent_query(self, agent_type: str, sample_idx: int) -> None:
        """
        Validate agent type, sample query is valid.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :raise ValueError if agent_type or sample_idx invalid.
        """
        if agent_type not in TrackedObjectType._member_names_:
            raise ValueError(f"Invalid agent type: {agent_type}")
        if agent_type not in self.agents.keys():
            raise ValueError(f"Agent type: {agent_type} is unavailable!")
        if self.batch_size < sample_idx:
            raise ValueError(f"Requsted sample index {sample_idx} larger than batch size {self.batch_size}!")

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return (
            len(self.ego) > 0
            and all([len(agent) > 0 for agent in self.agents.values()])
            and all([len(self.ego) == len(agent) for agent in self.agents.values()])
            and len(self.ego[0]) > 0
            and all([len(agent[0]) > 0 for agent in self.agents.values()])
            and all([len(self.ego[0]) == len(agent[0]) > 0 for agent in self.agents.values()])
            and self.ego[0].shape[-1] == self.ego_state_dim()
            and all([agent[0].shape[-1] == self.agents_states_dim() for agent in self.agents.values()])
        )

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches.
        """
        return len(self.ego)

    @classmethod
    def collate(cls, batch: List[GenericAgents]) -> GenericAgents:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        agents: Dict[str, List[FeatureDataType]] = defaultdict(list)
        for sample in batch:
            for agent_name, agent in sample.agents.items():
                agents[agent_name] += [agent[0]]
        return GenericAgents(ego=[item.ego[0] for item in batch], agents=agents)

    def to_feature_tensor(self) -> GenericAgents:
        """Implemented. See interface."""
        return GenericAgents(
            ego=[to_tensor(sample) for sample in self.ego],
            agents={agent_name: [to_tensor(sample) for sample in agent] for agent_name, agent in self.agents.items()},
        )

    def to_device(self, device: torch.device) -> GenericAgents:
        """Implemented. See interface."""
        return GenericAgents(
            ego=[to_tensor(ego).to(device=device) for ego in self.ego],
            agents={
                agent_name: [to_tensor(sample).to(device=device) for sample in agent]
                for agent_name, agent in self.agents.items()
            },
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> GenericAgents:
        """Implemented. See interface."""
        return GenericAgents(ego=data["ego"], agents=data["agents"])

    def unpack(self) -> List[GenericAgents]:
        """Implemented. See interface."""
        return [
            GenericAgents(
                ego=[self.ego[sample_idx]],
                agents={agent_name: [agent[sample_idx]] for agent_name, agent in self.agents.items()},
            )
            for sample_idx in range(self.batch_size)
        ]

    def num_agents_in_sample(self, agent_type: str, sample_idx: int) -> int:
        """
        Returns the number of agents at a given batch for given agent feature type.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: number of agents in the given batch.
        """
        self._validate_agent_query(agent_type, sample_idx)
        return self.agents[agent_type][sample_idx].shape[1]  # type: ignore

    @staticmethod
    def ego_state_dim() -> int:
        """
        :return: ego state dimension.
        """
        return GenericEgoFeatureIndex.dim()

    @staticmethod
    def agents_states_dim() -> int:
        """
        :return: agent state dimension.
        """
        return GenericAgentFeatureIndex.dim()

    @property
    def num_frames(self) -> int:
        """
        :return: number of frames.
        """
        return int(self.ego[0].shape[0])

    @property
    def ego_feature_dim(self) -> int:
        """
        :return: ego feature dimension.
        """
        return GenericAgents.ego_state_dim() * self.num_frames

    @property
    def agents_features_dim(self) -> int:
        """
        :return: ego feature dimension.
        """
        return GenericAgents.agents_states_dim() * self.num_frames

    def has_agents(self, agent_type: str, sample_idx: int) -> bool:
        """
        Check whether agents of specified type exist in the feature.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: whether agents exist in the feature.
        """
        self._validate_agent_query(agent_type, sample_idx)
        return self.num_agents_in_sample(agent_type, sample_idx) > 0

    def agent_processing_by_type(
        self, processing_function: Callable[[str, int], FeatureDataType], sample_idx: int
    ) -> FeatureDataType:
        """
        Apply agent processing functions across all agent types in features for given batch sample.
        :param processing_function: function to apply across agent types
        :param sample_idx: the batch index of interest.
        :return Processed agent feature across agent types.
        """
        agents: List[FeatureDataType] = []
        for agent_type in self.agents.keys():
            if self.has_agents(agent_type, sample_idx):
                agents.append(processing_function(agent_type, sample_idx))
        if len(agents) == 0:
            if isinstance(self.ego[sample_idx], torch.Tensor):
                return torch.empty(
                    (0, len(self.agents.keys()) * self.num_frames * GenericAgentFeatureIndex.dim()),
                    dtype=self.ego[sample_idx].dtype,
                    device=self.ego[sample_idx].device,
                )
            else:
                return np.empty(
                    (0, len(self.agents.keys()) * self.num_frames * GenericAgentFeatureIndex.dim()),
                    dtype=self.ego[sample_idx].dtype,
                )
        elif isinstance(agents[0], torch.Tensor):
            return torch.cat(agents, dim=0)
        else:
            return np.concatenate(agents, axis=0)

    def get_flatten_agents_features_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Flatten agents' features of specified type by stacking the agents' states along the num_frame dimension
        <np.ndarray: num_frames, num_agents, 8>] -> <np.ndarray: num_agents, num_frames x 8>].

        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, num_frames x 8>] agent feature.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.num_agents_in_sample(agent_type, sample_idx) == 0:
            if isinstance(self.ego[sample_idx], torch.Tensor):
                return torch.empty(
                    (0, self.num_frames * GenericAgentFeatureIndex.dim()),
                    dtype=self.ego[sample_idx].dtype,
                    device=self.ego[sample_idx].device,
                )
            else:
                return np.empty(
                    (0, self.num_frames * GenericAgentFeatureIndex.dim()),
                    dtype=self.ego[sample_idx].dtype,
                )

        data = self.agents[agent_type][sample_idx]
        axes = (1, 0) if isinstance(data, torch.Tensor) else (1, 0, 2)
        return data.transpose(*axes).reshape(data.shape[1], -1)

    def get_flatten_agents_features_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Flatten agents' features of all types by stacking the agents' states along the num_frame dimension
        <np.ndarray: num_frames, num_agents, 8>] -> <np.ndarray: num_agents, num_frames x 8>].

        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents, num_frames x 8>] agent feature.
        """
        return self.agent_processing_by_type(self.get_flatten_agents_features_by_type_in_sample, sample_idx)

    def get_present_ego_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present ego in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: 8>. ego at sample index.
        """
        self._validate_ego_query(sample_idx)
        return self.ego[sample_idx][-1]

    def get_present_agents_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Return the present agents of specified type in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, 8>. all agents at sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")
        return self.agents[agent_type][sample_idx][-1]

    def get_present_agents_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present agents of all types in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents, 8>. all agents at sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        return self.agent_processing_by_type(self.get_present_agents_by_type_in_sample, sample_idx)

    def get_ego_agents_center_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return ego center in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: 2>. (x, y) positions of the ego's center at sample index.
        """
        self._validate_ego_query(sample_idx)
        return self.get_present_ego_in_sample(sample_idx)[: GenericEgoFeatureIndex.y() + 1]

    def get_agents_centers_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's centers in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, 2>. (x, y) positions of the agents' centers at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")
        return self.get_present_agents_by_type_in_sample(agent_type, sample_idx)[:, : GenericAgentFeatureIndex.y() + 1]

    def get_agents_centers_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of all types' centers in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents, 2>.
            (x, y) positions of the agents' centers at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        return self.agent_processing_by_type(self.get_agents_centers_by_type_in_sample, sample_idx)

    def get_agents_length_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's length at the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents>. lengths of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")
        return self.get_present_agents_by_type_in_sample(agent_type, sample_idx)[:, GenericAgentFeatureIndex.length()]

    def get_agents_length_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of all types' length at the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents>. lengths of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        return self.agent_processing_by_type(self.get_agents_length_by_type_in_sample, sample_idx)

    def get_agents_width_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's width in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents>. width of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")
        return self.get_present_agents_by_type_in_sample(agent_type, sample_idx)[:, GenericAgentFeatureIndex.width()]

    def get_agents_width_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of all types' width in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents>. width of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty
        """
        return self.agent_processing_by_type(self.get_agents_width_by_type_in_sample, sample_idx)

    def get_agent_corners_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's corners in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, 4, 3>. (x, y, 1) positions of all the agents' corners at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")
        widths = self.get_agents_width_by_type_in_sample(agent_type, sample_idx)
        lengths = self.get_agents_length_by_type_in_sample(agent_type, sample_idx)

        half_widths = widths / 2.0
        half_lengths = lengths / 2.0

        feature_cls = np.array if isinstance(widths, np.ndarray) else torch.Tensor

        return feature_cls(
            [
                [
                    [half_length, half_width, 1.0],
                    [-half_length, half_width, 1.0],
                    [-half_length, -half_width, 1.0],
                    [half_length, -half_width, 1.0],
                ]
                for half_width, half_length in zip(half_widths, half_lengths)
            ]
        )

    def get_agent_corners_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of all types' corners in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents, 4, 3>.
            (x, y, 1) positions of all the agents' corners at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        return self.agent_processing_by_type(self.get_agent_corners_by_type_in_sample, sample_idx)


class GenericEgoFeatureIndex:
    """
    A convenience class for assigning semantic meaning to the tensor index
        in the final output ego feature.

    It is intended to be used like an IntEnum, but supported by TorchScript.
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the x coordinate of the ego.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y coordinate of the ego.
        :return: index
        """
        return 1

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the heading of the ego.
        :return: index
        """
        return 2

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the x velocity of the ego.
        :return: index
        """
        return 3

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the y velocity of the ego.
        :return: index
        """
        return 4

    @staticmethod
    def ax() -> int:
        """
        The dimension corresponding to the x acceleration of the ego.
        :return: index
        """
        return 5

    @staticmethod
    def ay() -> int:
        """
        The dimension corresponding to the y acceleration of the ego.
        :return: index
        """
        return 6

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the EgoFeature.
        :return: number of features.
        """
        return 7


class GenericAgentFeatureIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
        in the final output agents feature.

    It is intended to be used like an IntEnum, but supported by TorchScript.
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the x coordinate of the agent.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y coordinate of the agent.
        :return: index
        """
        return 1

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the heading of the agent.
        :return: index
        """
        return 2

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the x velocity of the agent.
        :return: index
        """
        return 3

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the y velocity of the agent.
        :return: index
        """
        return 4

    @staticmethod
    def yaw_rate() -> int:
        """
        The dimension corresponding to the yaw rate of the agent.
        :return: index
        """
        return 5

    @staticmethod
    def length() -> int:
        """
        The dimension corresponding to the length of the agent.
        :return: index
        """
        return 6

    @staticmethod
    def width() -> int:
        """
        The dimension corresponding to the width of the agent.
        :return: index
        """
        return 7

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the AgentsFeature.
        :return: number of features.
        """
        return 8
