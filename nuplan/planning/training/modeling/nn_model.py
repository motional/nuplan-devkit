from typing import List

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


class NNModule(torch.nn.Module):

    def __init__(self,
                 future_trajectory_sampling: TrajectorySampling,
                 feature_builders: List[AbstractFeatureBuilder],
                 target_builders: List[AbstractTargetBuilder]):
        """
        Construct a model with feature and target builders
        :param future_trajectory_sampling: parameters for a predicted trajectory
        :param feature_builders: the list of builders which will compute features for this model
        :param target_builders: the list of builders which will compute targets for this model
        """
        super().__init__()

        self.future_trajectory_sampling = future_trajectory_sampling
        self.feature_builders = feature_builders
        self.target_builders = target_builders

    def get_list_of_required_feature(self) -> List[AbstractFeatureBuilder]:
        """
        :return list of required input features to the model
        """
        return self.feature_builders

    def get_list_of_computed_target(self) -> List[AbstractTargetBuilder]:
        """
        :return list of features that the model computes
        """
        return self.target_builders
