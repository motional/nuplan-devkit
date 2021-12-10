from typing import cast

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.lanegcn_utils import Actor2ActorAttention, Lane2ActorAttention, LaneNet, \
    LinearWithGroupNorm
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


def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


class LaneGCN(NNModule):

    def __init__(self,
                 map_net_scales: int,
                 num_res_blocks: int,
                 num_attention_layers: int,
                 a2a_dist_threshold: float,
                 l2a_dist_threshold: float,
                 num_output_features: int,
                 feature_dim: int,
                 # Parameters for Features
                 vector_map_feature_radius: int,
                 past_trajectory_sampling: TrajectorySampling,
                 # Parameters for Targets
                 future_trajectory_sampling: TrajectorySampling):
        """
        :param map_net_scales: Number of scales to extend the predecessor and successor lane nodes.
        :param num_res_blocks: Number of residual blocks for the GCN (LaneGCN uses 4).
        :param num_attention_layers: Number of times to repeatedly apply the attention layer.
        :param a2a_dist_threshold: [m] distance threshold for aggregating actor-to-actor nodes
        :param l2a_dist_threshold: [m] distance threshold for aggregating map-to-actor nodes
        :param num_output_features: number of target features
        :param feature_dim: hidden layer dimension
        :param vector_map_feature_radius: The query radius scope relative to the current ego-pose.
        :param past_trajectory_sampling: Sampling parameters for past trajectory
        :param future_trajectory_sampling: Sampling parameters for future trajectory
        """
        super().__init__(feature_builders=[VectorMapFeatureBuilder(radius=vector_map_feature_radius),
                                           AgentsFeatureBuilder(trajectory_sampling=past_trajectory_sampling)],
                         target_builders=[
                             EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_sampling)],
                         future_trajectory_sampling=future_trajectory_sampling)

        # LaneGCN components
        self.feature_dim = feature_dim
        self.lane_net = LaneNet(
            lane_input_len=2,
            lane_feature_len=self.feature_dim,
            num_scales=map_net_scales,
            num_res_blocks=num_res_blocks,
            is_map_feat=False,
        )
        self.ego_feature_extractor = torch.nn.Sequential(
            nn.Linear((past_trajectory_sampling.num_poses + 1) * AgentsFeature.ego_state_dim(), self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            LinearWithGroupNorm(self.feature_dim, self.feature_dim, num_groups=1, activation=False),
        )
        self.agent_feature_extractor = torch.nn.Sequential(
            nn.Linear((past_trajectory_sampling.num_poses + 1) * AgentsFeature.agents_states_dim(), self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            LinearWithGroupNorm(self.feature_dim, self.feature_dim, num_groups=1, activation=False),
        )
        self.lane2actor_attention = Lane2ActorAttention(
            lane_feature_len=self.feature_dim,
            actor_feature_len=self.feature_dim,
            num_attention_layers=num_attention_layers,
            dist_threshold_m=l2a_dist_threshold
        )
        self.actor2actor_attention = Actor2ActorAttention(
            actor_feature_len=self.feature_dim,
            num_attention_layers=num_attention_layers,
            dist_threshold_m=a2a_dist_threshold,
        )
        # Final mlp
        self._mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, num_output_features),
        )

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
        ego_agent_features = cast(AgentsFeature, features["agents"])
        ego_past_trajectory = ego_agent_features.ego  # batch_size x num_frames x 3

        # Extract batches
        batch_size = ego_agent_features.batch_size

        # Extract features
        ego_features = []
        # map and agent feature have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            lane_graph_input = {}
            lane_graph_input['coords'] = vector_map_data.coords[sample_idx]
            lane_graph_input['connections'] = vector_map_data.multi_scale_connections[sample_idx][1]
            lane_features = self.lane_net(lane_graph_input)
            lane_centers = lane_graph_input['coords'].mean(axis=1)

            sample_ego_feature = self.ego_feature_extractor(ego_past_trajectory[sample_idx].view(1, -1))
            sample_ego_center = ego_agent_features.get_ego_agents_center_in_sample(sample_idx)

            if ego_agent_features.has_agents(sample_idx):
                sample_agents_feature = self.agent_feature_extractor(
                    ego_agent_features.get_flatten_agents_features_in_sample(sample_idx)
                )
                sample_agents_center = ego_agent_features.get_agents_centers_in_sample(sample_idx)
                ego_agents_feature = torch.cat([sample_ego_feature, sample_agents_feature], dim=0)
                ego_agents_center = torch.cat([sample_ego_center.unsqueeze(dim=0), sample_agents_center], dim=0)
            else:
                ego_agents_feature = sample_ego_feature
                ego_agents_center = sample_ego_center.unsqueeze(dim=0)

            ego_agents_feature = self.lane2actor_attention(lane_features, lane_centers, ego_agents_feature,
                                                           ego_agents_center)
            ego_agents_feature = self.actor2actor_attention(ego_agents_feature, ego_agents_center)
            ego_features.append(ego_agents_feature[0])

        ego_features = torch.cat(ego_features).view(batch_size, -1)

        # Predict future
        predictions = self._mlp(ego_features)

        return {"trajectory": Trajectory(data=convert_predictions_to_trajectory(predictions))}
