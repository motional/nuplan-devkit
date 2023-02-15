from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


@dataclass
class Trajectories(AbstractModelFeature):
    """
    A feature that contains multiple trajectories
    """

    trajectories: List[Trajectory]

    def to_feature_tensor(self) -> Trajectories:
        """Implemented. See interface."""
        return Trajectories(trajectories=[trajectory.to_feature_tensor() for trajectory in self.trajectories])

    def to_device(self, device: torch.device) -> Trajectories:
        """Implemented. See interface."""
        return Trajectories(trajectories=[trajectory.to_device(device) for trajectory in self.trajectories])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Trajectories:
        """Implemented. See interface."""
        return Trajectories(trajectories=[Trajectory.deserialize(trajectory) for trajectory in data["trajectories"]])

    @property
    def number_of_trajectories(self) -> int:
        """
        :return: number of trajectories in this feature.
        """
        return len(self.trajectories)

    def unpack(self) -> List[Trajectories]:
        """Implemented. See interface."""
        return [Trajectories([trajectories]) for trajectories in self.trajectories]
