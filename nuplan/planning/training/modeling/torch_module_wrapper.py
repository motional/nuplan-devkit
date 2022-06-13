import abc
from typing import List

import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


class TorchModuleWrapper(torch.nn.Module):
    """Torch module wrapper that encapsulates builders for constructing model features and targets."""

    def __init__(
        self,
        future_trajectory_sampling: TrajectorySampling,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ):
        """
        Construct a model with feature and target builders.
        :param future_trajectory_sampling: Parameters for a predicted trajectory.
        :param feature_builders: The list of builders which will compute features for this model.
        :param target_builders: The list of builders which will compute targets for this model.
        """
        super().__init__()

        self.future_trajectory_sampling = future_trajectory_sampling
        self.feature_builders = feature_builders
        self.target_builders = target_builders

    def get_list_of_required_feature(self) -> List[AbstractFeatureBuilder]:
        """Get list of required input features to the model."""
        return self.feature_builders

    def get_list_of_computed_target(self) -> List[AbstractTargetBuilder]:
        """Get list of features that the model computes."""
        return self.target_builders

    @abc.abstractmethod
    def forward(self, features: FeaturesType) -> TargetsType:
        """
        The main inference call for the model.
        :param features: A dictionary of the required features.
        :return: The results of the inference as a TargetsType.
        """
        pass
