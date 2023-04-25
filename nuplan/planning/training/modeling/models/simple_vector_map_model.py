from typing import Dict, List, Tuple, cast

import torch
from torch import nn

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.scriptable_torch_module_wrapper import ScriptableTorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)


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


def convert_predictions_to_trajectory(predictions: torch.Tensor, trajectory_state_size: int) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :param trajectory_state_size: trajectory state size
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.reshape(num_batches, -1, trajectory_state_size)


class VectorMapSimpleMLP(ScriptableTorchModuleWrapper):
    """Simple vector-based model that encodes agents and map elements through an MLP."""

    def __init__(
        self,
        num_output_features: int,
        hidden_size: int,
        vector_map_feature_radius: int,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
    ):
        """
        Initialize the simple vector map model.
        :param num_output_features: number of target features
        :param hidden_size: size of hidden layers of MLP
        :param vector_map_feature_radius: The query radius scope relative to the current ego-pose.
        :param past_trajectory_sampling: Sampling parameters for past trajectory
        :param future_trajectory_sampling: Sampling parameters for future trajectory
        """
        super().__init__(
            feature_builders=[
                VectorMapFeatureBuilder(radius=vector_map_feature_radius),
                AgentsFeatureBuilder(past_trajectory_sampling),
            ],
            target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling)],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        self._hidden_size = hidden_size

        # Vectormap feature input size is 2D start lane coord + 2D end lane coord
        self.vectormap_mlp = create_mlp(
            input_size=2 * VectorMap.lane_coord_dim(), output_size=self._hidden_size, hidden_size=self._hidden_size
        )

        # Ego trajectory feature
        self.ego_mlp = create_mlp(
            input_size=(past_trajectory_sampling.num_poses + 1) * Agents.ego_state_dim(),
            output_size=self._hidden_size,
            hidden_size=self._hidden_size,
        )

        # Agent trajectory feature
        self._agent_mlp_dim = (past_trajectory_sampling.num_poses + 1) * Agents.agents_states_dim()
        self.agent_mlp = create_mlp(
            input_size=self._agent_mlp_dim,
            output_size=self._hidden_size,
            hidden_size=self._hidden_size,
        )

        # Final mlp
        self._mlp = create_mlp(
            input_size=3 * self._hidden_size, output_size=num_output_features, hidden_size=self._hidden_size
        )

        self._vector_map_flatten_lane_coord_dim = VectorMap.flatten_lane_coord_dim()
        self._trajectory_state_size = Trajectory.state_size()

    @torch.jit.unused
    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_map": VectorMap,
                            "agents": Agents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        # Recover features
        vector_map_data = cast(VectorMap, features["vector_map"])
        ego_agents_feature = cast(Agents, features["agents"])

        # Pack the data in the same manner as it was created in the `scriptable_forward()` methods
        #  of the feature builders.
        # Currently, the feature builders for this model output only lists of tensors.
        #  The empty dictionaries are required to satisfy the interface.
        tensor_inputs: Dict[str, torch.Tensor] = {}

        list_tensor_inputs = {
            "vector_map.coords": vector_map_data.coords,
            "agents.ego": ego_agents_feature.ego,
            "agents.agents": ego_agents_feature.agents,
        }

        list_list_tensor_inputs: Dict[str, List[List[torch.Tensor]]] = {}

        # Run the core logic of the model
        output_tensors, output_list_tensors, output_list_list_tensors = self.scriptable_forward(
            tensor_inputs, list_tensor_inputs, list_list_tensor_inputs
        )

        # Unpack the model's output
        return {"trajectory": Trajectory(data=output_tensors["trajectory"])}

    @torch.jit.export
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        # Extract data
        ego_past_trajectory = list_tensor_data["agents.ego"]
        ego_agents_agents = list_tensor_data["agents.agents"]
        vector_map_coords = list_tensor_data["vector_map.coords"]

        # Extract batches
        if len(vector_map_coords) != len(ego_agents_agents) or len(vector_map_coords) != len(ego_past_trajectory):
            raise ValueError(
                f"Mixed batch sizes passed to scriptable_forward: vector_map.coords = {len(vector_map_coords)}, agents.agents = {len(ego_agents_agents)}, agents.ego_past_trajectory={len(ego_past_trajectory)}"
            )
        batch_size = len(vector_map_coords)

        # Extract features
        vector_map_feature: List[torch.Tensor] = []
        agents_feature: List[torch.Tensor] = []
        ego_feature: List[torch.Tensor] = []

        # map and agent feature have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            sample_ego_feature = self.ego_mlp(ego_past_trajectory[sample_idx].view(1, -1))
            ego_feature.append(torch.max(sample_ego_feature, dim=0).values)

            # Use reshape() throughout insead of view() because incoming tensors may not be contiguous
            #  (e.g. if being called from torchscript)
            vectormap_coords = vector_map_coords[sample_idx].reshape(-1, self._vector_map_flatten_lane_coord_dim)

            # Handle the case of zero-length vector map features.
            if vectormap_coords.numel() == 0:
                vectormap_coords = torch.zeros(
                    (1, self._vector_map_flatten_lane_coord_dim),
                    dtype=vectormap_coords.dtype,
                    device=vectormap_coords.device,
                )

            sample_vectormap_feature = self.vectormap_mlp(vectormap_coords)
            vector_map_feature.append(torch.max(sample_vectormap_feature, dim=0).values)

            this_agents_feature = ego_agents_agents[sample_idx]

            # We always need to run the agent_mlp during distributed training.
            # Otherwise, training crashes with "RuntimeError: Expected to finish reduction
            #    in the prior iteration before starting another one
            #
            # When agents are not present, use this variable to detect and mask the feature.
            agents_multiplier = float(min(this_agents_feature.shape[1], 1))

            if this_agents_feature.shape[1] > 0:  # if there exist at least one valid agent in the sample
                # Flatten agents' features by stacking along the frame dimension.
                # Go from <num_frames, num_agents, feature_dim> -> <num_agents, num_frames * num_features
                orig_shape = this_agents_feature.shape
                flattened_agents = this_agents_feature.transpose(1, 0).reshape(orig_shape[1], -1)
            else:
                flattened_agents = torch.zeros(
                    (this_agents_feature.shape[0], self._agent_mlp_dim),
                    device=sample_vectormap_feature.device,
                    dtype=sample_vectormap_feature.dtype,
                    layout=sample_vectormap_feature.layout,
                )

            sample_agent_feature = self.agent_mlp(flattened_agents)
            sample_agent_feature *= agents_multiplier
            agents_feature.append(torch.max(sample_agent_feature, dim=0).values)

        vector_map_feature = torch.cat(vector_map_feature).reshape(batch_size, -1)
        ego_feature = torch.cat(ego_feature).reshape(batch_size, -1)
        agents_feature = torch.cat(agents_feature).reshape(batch_size, -1)

        input_features = torch.cat([vector_map_feature, ego_feature, agents_feature], dim=1)

        # Predict future
        predictions = self._mlp(input_features)

        output_tensors: Dict[str, torch.Tensor] = {
            "trajectory": convert_predictions_to_trajectory(predictions, self._trajectory_state_size)
        }

        output_list_tensors: Dict[str, List[torch.Tensor]] = {}
        output_list_list_tensors: Dict[str, List[List[torch.Tensor]]] = {}

        return output_tensors, output_list_tensors, output_list_list_tensors
