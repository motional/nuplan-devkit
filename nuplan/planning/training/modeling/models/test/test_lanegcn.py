import os
import socket
import unittest
from typing import Callable

import torch
from munch import Munch
from torch.nn.parallel import DistributedDataParallel as DDP

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.lanegcn_model import LaneGCN
from nuplan.planning.training.modeling.models.lanegcn_utils import (
    Actor2ActorAttention,
    Actor2LaneAttention,
    GraphAttention,
    Lane2ActorAttention,
    LaneNet,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap


class TestGraphAttention(unittest.TestCase):
    """Test graph attention layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.src_feature_len = 4
        self.dst_feature_len = 4
        self.dist_threshold = 6.0
        self.model = GraphAttention(self.src_feature_len, self.dst_feature_len, self.dist_threshold)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_src_nodes = 2
        num_dst_nodes = 3
        src_node_features = torch.zeros((num_src_nodes, self.src_feature_len))
        src_node_pos = torch.zeros((num_src_nodes, 2))
        dst_node_features = torch.zeros((num_dst_nodes, self.dst_feature_len))
        dst_node_pos = torch.zeros((num_dst_nodes, 2))

        output = self.model.forward(src_node_features, src_node_pos, dst_node_features, dst_node_pos)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_dst_nodes, self.dst_feature_len))


class TestActor2ActorAttention(unittest.TestCase):
    """Test actor-to-actor attention layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.actor_feature_len = 4
        self.num_attention_layers = 2
        self.dist_threshold_m = 6.0

        self.model = Actor2ActorAttention(self.actor_feature_len, self.num_attention_layers, self.dist_threshold_m)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_actors = 3
        actor_features = torch.zeros((num_actors, self.actor_feature_len))
        actor_centers = torch.zeros((num_actors, 2))

        output = self.model.forward(actor_features, actor_centers)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_actors, self.actor_feature_len))


class TestLane2ActorAttention(unittest.TestCase):
    """Test lane-to-actor attention layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.lane_feature_len = 4
        self.actor_feature_len = 4
        self.num_attention_layers = 2
        self.dist_threshold_m = 6.0

        self.model = Lane2ActorAttention(
            self.lane_feature_len, self.actor_feature_len, self.num_attention_layers, self.dist_threshold_m
        )

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_lanes = 2
        num_actors = 3
        lane_features = torch.zeros((num_lanes, self.lane_feature_len))
        lane_centers = torch.zeros((num_lanes, 2))
        actor_features = torch.zeros((num_actors, self.actor_feature_len))
        actor_centers = torch.zeros((num_actors, 2))

        output = self.model.forward(lane_features, lane_centers, actor_features, actor_centers)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_actors, self.actor_feature_len))


class TestActor2LaneAttention(unittest.TestCase):
    """Test actor-to-lane attention layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.lane_feature_len = 4
        self.actor_feature_len = 4
        self.num_attention_layers = 2
        self.dist_threshold_m = 6.0

        self.model = Actor2LaneAttention(
            self.actor_feature_len, self.lane_feature_len, self.num_attention_layers, self.dist_threshold_m
        )

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_lanes = 2
        num_actors = 3
        meta_info_len = 6
        lane_features = torch.zeros((num_lanes, self.lane_feature_len))
        lane_meta = torch.zeros((num_lanes, meta_info_len))
        lane_centers = torch.zeros((num_lanes, 2))
        actor_features = torch.zeros((num_actors, self.actor_feature_len))
        actor_centers = torch.zeros((num_actors, 2))

        output = self.model.forward(actor_features, actor_centers, lane_features, lane_meta, lane_centers)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_lanes, self.actor_feature_len))


class TestLaneNet(unittest.TestCase):
    """Test lane net layer."""

    def setUp(self) -> None:
        """Set up the test."""
        self.lane_input_len = 2
        self.lane_feature_len = 4
        self.num_scales = 2
        self.num_res_blocks = 3
        self.model = LaneNet(
            lane_input_len=self.lane_input_len,
            lane_feature_len=self.lane_feature_len,
            num_scales=self.num_scales,
            num_residual_blocks=self.num_res_blocks,
            is_map_feat=False,
        )

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_lanes = 4
        lane_input = torch.zeros((num_lanes, self.lane_input_len, 2))
        multi_scale_connections = {
            1: torch.tensor([[0, 1], [1, 2], [2, 3]]),
            2: torch.tensor([[0, 2], [1, 3]]),
        }
        vector_map = Munch(multi_scale_connections=multi_scale_connections)

        output = self.model.forward(lane_input, vector_map.multi_scale_connections)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_lanes, self.lane_feature_len))


class TestLaneGCN(unittest.TestCase):
    """Test LaneGCN model."""

    def _build_model(self) -> LaneGCN:
        """
        Creates a new instance of a LaneGCN with some default parameters.
        """
        model = LaneGCN(
            map_net_scales=4,
            num_res_blocks=4,
            num_attention_layers=5,
            a2a_dist_threshold=20,
            l2a_dist_threshold=20,
            num_output_features=12,
            feature_dim=32,
            vector_map_feature_radius=30,
            vector_map_connection_scales=[1, 2, 3, 4],
            past_trajectory_sampling=TrajectorySampling(num_poses=4, time_horizon=1.5),
            future_trajectory_sampling=TrajectorySampling(num_poses=12, time_horizon=6),
        )

        return model

    def _build_input_features(self, device: torch.device, include_agents: bool, include_lanes: bool) -> FeaturesType:
        # TODO: factor out this into a common method between the LaneGCN and SimpleVectorMapMLP
        """
        Creates a set of input features for use with unit testing.
        :param device: The device on which to create the tensors.
        :param include_agents: If true, the generated input features will have agents.
            If not, then there will be no agents in the agents feature.
        :param include_lanes: If true, the generated input features will have lanes.
            If not, then there will be no lanes in the vectormap feature.
        :return: FeaturesType to be consumed by the model
        """
        # Numbers chosen arbitrarily
        num_frames = 5
        num_coords = 1000
        num_groupings = 100
        num_multi_scale_connections = 800
        num_lanes = num_coords if include_lanes else 0
        num_connections = num_multi_scale_connections if include_lanes else 0
        num_agents = num_frames if include_agents else 0

        vector_map_coords = [
            torch.zeros(
                (num_lanes, VectorMap.lane_coord_dim(), VectorMap.lane_coord_dim()), dtype=torch.float32, device=device
            )
        ]
        vector_map_lane_groupings = [[torch.zeros((num_groupings), device=device)]]
        multi_scale_connections = [
            {
                1: torch.zeros((num_connections, 2), device=device).long(),
                2: torch.zeros((num_connections, 2), device=device).long(),
                3: torch.zeros((num_connections, 2), device=device).long(),
                4: torch.zeros((num_connections, 2), device=device).long(),
            }
        ]
        on_route_status = [torch.zeros((num_lanes, VectorMap.on_route_status_encoding_dim()), device=device)]
        traffic_light_data = [torch.zeros((num_lanes, 4), device=device)]

        vector_map_feature = VectorMap(
            coords=vector_map_coords,
            lane_groupings=vector_map_lane_groupings,
            multi_scale_connections=multi_scale_connections,
            on_route_status=on_route_status,
            traffic_light_data=traffic_light_data,
        )

        ego_agents = [torch.zeros((num_frames, Agents.ego_state_dim()), dtype=torch.float32, device=device)]
        agent_agents = [
            torch.zeros((num_frames, num_agents, Agents.agents_states_dim()), dtype=torch.float32, device=device)
        ]

        agents_feature = Agents(ego=ego_agents, agents=agent_agents)

        return {
            "vector_map": vector_map_feature,
            "agents": agents_feature,
        }

    def _find_free_port(self) -> int:
        """
        Finds a free port to use for gloo server.
        :return: A port not in use.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # passing "0" as port will instruct the OS to pick an open port at random.
            s.bind(("localhost", 0))
            address, port = s.getsockname()
            return int(port)

    def _init_distributed_process_group(self) -> None:
        """
        Sets up the torch distributed processing server.
        :param port: The port to use for the gloo server.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self._find_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        torch.distributed.init_process_group(backend="gloo")

    def _assert_valid_output(self, model_output: TargetsType) -> None:
        """
        Validates that the output from the model has the correct keys and that the tensor is of the correct type.
        :param model_output: The output from the model.
        """
        self.assertTrue("trajectory" in model_output)
        self.assertTrue(isinstance(model_output["trajectory"], Trajectory))

        predicted_trajectory: Trajectory = model_output["trajectory"]
        self.assertIsNotNone(predicted_trajectory.data)

    def _perform_backprop_step(
        self,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        predictions: TargetsType,
    ) -> None:
        """
        Performs a backpropagation step.
        :param optimizer: The optimizer to use for training.
        :param loss_function: The loss function to use.
        :param predictions: The output from the model.
        """
        loss = loss_function(predictions["trajectory"].data, torch.zeros_like(predictions["trajectory"].data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_backprop(self) -> None:
        """
        Tests that the LaneGCN model can train with DDP.
        This test was developed in response to an error related to zero agent input.
        """
        self._init_distributed_process_group()

        device = torch.device("cpu")

        model = self._build_model().to(device)
        ddp_model = DDP(model, device_ids=None, output_device=None)
        optimizer = torch.optim.RMSprop(ddp_model.parameters())
        loss_function = torch.nn.MSELoss()
        num_epochs = 3

        # Alternate batches containing many and zero agents.
        for _ in range(num_epochs):
            for include_agents in [True, False]:
                for include_lanes in [True, False]:
                    input_features = self._build_input_features(
                        device, include_agents=include_agents, include_lanes=include_lanes
                    )
                    predictions = ddp_model.forward(input_features)
                    self._assert_valid_output(predictions)
                    self._perform_backprop_step(optimizer, loss_function, predictions)


if __name__ == "__main__":
    unittest.main()
