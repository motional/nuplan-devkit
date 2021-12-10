from typing import cast

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.nn_model import NNModule
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import AgentsFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import \
    EgoTrajectoryTargetBuilder
from torch import nn


def create_mlp(input_size: int, output_size: int, hidden_size: int = 128) -> torch.nn.Module:
    """
    Create MLP
    :param input_size: input feature size
    :param output_size: output feature size
    :param hidden_size: hidden layer
    :return: sequential network
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


class VectorMapSimpleMLP(NNModule):

    def __init__(self,
                 num_output_features: int,
                 hidden_size: int,
                 # Parameters for Features
                 vector_map_feature_radius: int,
                 past_trajectory_sampling: TrajectorySampling,
                 # Parameters for Targets
                 future_trajectory_sampling: TrajectorySampling):
        """
        :param num_output_features: number of target features
        :param hidden_size: size of hidden layers of MLP
        :param vector_map_feature_radius: The query radius scope relative to the current ego-pose.
        :param past_trajectory_sampling: Sampling parameters for past trajectory
        :param future_trajectory_sampling: Sampling parameters for future trajectory
        """
        super().__init__(feature_builders=[VectorMapFeatureBuilder(radius=vector_map_feature_radius),
                                           AgentsFeatureBuilder(past_trajectory_sampling)],
                         target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling)],
                         future_trajectory_sampling=future_trajectory_sampling)

        self._hidden_size = hidden_size
        # Vectormap feature input size is 2D start lane coord + 2D end lane coord
        self.vectormap_mlp = create_mlp(input_size=2 * VectorMap.lane_coord_dim(),
                                        output_size=self._hidden_size,
                                        hidden_size=self._hidden_size)
        # Ego trajectory feature
        self.ego_mlp = create_mlp(
            input_size=(past_trajectory_sampling.num_poses + 1) * AgentsFeature.ego_state_dim(),
            output_size=self._hidden_size, hidden_size=self._hidden_size)
        # Agent trajectory feature
        self.agent_mlp = create_mlp(
            input_size=(past_trajectory_sampling.num_poses + 1) * AgentsFeature.agents_states_dim(),
            output_size=self._hidden_size, hidden_size=self._hidden_size)

        # Final mlp
        self._mlp = create_mlp(input_size=3 * self._hidden_size, output_size=num_output_features,
                               hidden_size=self._hidden_size)

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_map": VectorMap,
                            "agents": AgentsFeatures,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        # Recover features
        vector_map_data = cast(VectorMap, features["vector_map"])
        ego_agents_feature = cast(AgentsFeature, features["agents"])

        # Extract data
        ego_past_trajectory = ego_agents_feature.ego  # batch_size x num_frames x 3

        # Extract batches
        batch_size = ego_agents_feature.batch_size

        # Extract features
        vector_map_feature = []
        agents_feature = []
        ego_feature = []
        # map and agent feature have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            sample_vectormap_feature = self.vectormap_mlp(
                vector_map_data.coords[sample_idx].view(-1, VectorMap.flatten_lane_coord_dim()))
            vector_map_feature.append(torch.max(sample_vectormap_feature, dim=0).values)
            sample_ego_feature = self.ego_mlp(ego_past_trajectory[sample_idx].view(1, -1))
            ego_feature.append(torch.max(sample_ego_feature, dim=0).values)

            if ego_agents_feature.has_agents(sample_idx):  # if there exist at least one valid agent in the sample
                sample_agent_feature = self.agent_mlp(
                    ego_agents_feature.get_flatten_agents_features_in_sample(sample_idx))
                agents_feature.append(torch.max(sample_agent_feature, dim=0).values)
            else:
                agents_feature.append(torch.zeros_like(ego_feature[-1]))

        vector_map_feature = torch.cat(vector_map_feature).view(batch_size, -1)
        ego_feature = torch.cat(ego_feature).view(batch_size, -1)
        agents_feature = torch.cat(agents_feature).view(batch_size, -1)

        input_features = torch.cat([vector_map_feature, ego_feature, agents_feature], dim=1)

        # Predict future
        predictions = self._mlp(input_features)

        return {"trajectory": Trajectory(data=convert_predictions_to_trajectory(predictions))}
