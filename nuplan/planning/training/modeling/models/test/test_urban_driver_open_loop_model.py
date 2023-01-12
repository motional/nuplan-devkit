import os
import socket
import unittest
from typing import Callable, Dict, List

import torch
import torch.nn
import torch.optim
from torch.nn.parallel import DistributedDataParallel as DDP

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model import (
    UrbanDriverOpenLoopModel,
    UrbanDriverOpenLoopModelFeatureParams,
    UrbanDriverOpenLoopModelParams,
    UrbanDriverOpenLoopModelTargetParams,
)
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model_utils import (
    MLP,
    LocalMLP,
    LocalSubGraph,
    LocalSubGraphLayer,
    MultiheadAttentionGlobalHead,
    SinusoidalPositionalEmbedding,
    TypeEmbedding,
    pad_avails,
    pad_polylines,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap


class TestUrbanDriverOpenLoopUtils(unittest.TestCase):
    """Test UrbanDriverOpenLoop utils functions."""

    def test_pad_avails(self) -> None:
        """Test padding availability masks."""
        num_elements = 10
        num_points = 20
        input = torch.ones((num_elements, num_points), dtype=torch.bool)

        elements_padded = pad_avails(input, 20, 0)
        self.assertEqual(elements_padded.dtype, torch.bool)
        self.assertEqual(elements_padded.shape, (20, num_points))
        torch.testing.assert_allclose(elements_padded[10:, :], torch.zeros((10, num_points), dtype=torch.bool))

        points_padded = pad_avails(input, 30, 1)
        self.assertEqual(points_padded.dtype, torch.bool)
        self.assertEqual(points_padded.shape, (num_elements, 30))
        torch.testing.assert_allclose(points_padded[:, 20:], torch.zeros((num_elements, 10), dtype=torch.bool))

    def test_pad_polylines(self) -> None:
        """Test padding polyline features."""
        num_elements = 10
        num_points = 20
        num_features = 2
        input = torch.ones((num_elements, num_points, num_features), dtype=torch.float32)

        elements_padded = pad_polylines(input, 20, 0)
        self.assertEqual(elements_padded.dtype, torch.float32)
        self.assertEqual(elements_padded.shape, (20, num_points, num_features))
        torch.testing.assert_allclose(
            elements_padded[10:, :, :], torch.zeros((10, num_points, num_features), dtype=torch.float32)
        )

        points_padded = pad_polylines(input, 30, 1)
        self.assertEqual(points_padded.dtype, torch.float32)
        self.assertEqual(points_padded.shape, (num_elements, 30, num_features))
        torch.testing.assert_allclose(
            points_padded[:, 20:, :], torch.zeros((num_elements, 10, num_features), dtype=torch.float32)
        )

        features_padded = pad_polylines(input, 3, 2)
        self.assertEqual(features_padded.dtype, torch.float32)
        self.assertEqual(features_padded.shape, (num_elements, num_points, 3))
        torch.testing.assert_allclose(
            features_padded[:, :, 2:], torch.zeros((num_elements, num_points, 1), dtype=torch.float32)
        )


class TestLocalMLP(unittest.TestCase):
    """Test LocalMLP layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.dim_in = 256
        self.model = LocalMLP(self.dim_in)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_inputs = 10
        inputs = torch.zeros((num_inputs, self.dim_in))

        output = self.model.forward(inputs)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_inputs, self.dim_in))


class TestMLP(unittest.TestCase):
    """Test MLP layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.input_dim = 256
        self.hidden_dim = 256 * 4
        self.output_dim = 12 * 3
        self.num_layers = 3
        self.model = MLP(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_inputs = 10
        inputs = torch.zeros((num_inputs, self.input_dim))

        output = self.model.forward(inputs)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_inputs, self.output_dim))


class TestSinusoidalPositionalEmbedding(unittest.TestCase):
    """Test SinusoidalPositionalEmbedding layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.embedding_size = 256
        self.model = SinusoidalPositionalEmbedding(self.embedding_size)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        batch_size = 2
        num_elements = 10
        num_points = 20
        inputs = torch.zeros((batch_size, num_elements, num_points, self.embedding_size))

        output = self.model.forward(inputs)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_points, 1, self.embedding_size))


class TestTypeEmbedding(unittest.TestCase):
    """Test TypeEmbedding layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.embedding_dim = 256
        self.feature_types = {
            'NONE': -1,
            'EGO': 0,
            'VEHICLE': 1,
            'BICYCLE': 2,
            'PEDESTRIAN': 3,
            'LANE': 4,
            'STOP_LINE': 5,
            'CROSSWALK': 6,
            'LEFT_BOUNDARY': 7,
            'RIGHT_BOUNDARY': 8,
            'ROUTE_LANES': 9,
        }
        self.model = TypeEmbedding(self.embedding_dim, self.feature_types)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        device = torch.device("cpu")
        batch_size = 2
        max_agents = 30
        agent_features = ['VEHICLE', 'BICYCLE', 'PEDESTRIAN']
        map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        max_elements = {
            'LANE': 30,
            'LEFT_BOUNDARY': 30,
            'RIGHT_BOUNDARY': 30,
            'STOP_LINE': 20,
            'CROSSWALK': 20,
            'ROUTE_LANES': 30,
        }
        num_elements = 1 + max_agents * len(agent_features) + sum(max_elements.values())

        output = self.model.forward(batch_size, max_agents, agent_features, map_features, max_elements, device)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, num_elements, self.embedding_dim))


class TestLocalSubGraphLayer(unittest.TestCase):
    """Test LocalSubGraphLayer layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.dim_in = 256
        self.dim_out = 256
        self.model = LocalSubGraphLayer(self.dim_in, self.dim_out)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_elements = 10
        num_points = 20
        inputs = torch.zeros((num_elements, num_points, self.dim_in))
        invalid_mask = torch.zeros((num_elements, num_points), dtype=torch.bool)

        output = self.model.forward(inputs, invalid_mask)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_elements, num_points, self.dim_out))


class TestLocalSubGraph(unittest.TestCase):
    """Test LocalSubGraph layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.num_layers = 3
        self.dim_in = 256
        self.model = LocalSubGraph(self.num_layers, self.dim_in)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        batch_size = 2
        num_elements = 10
        num_points = 20
        inputs = torch.zeros((batch_size, num_elements, num_points, self.dim_in), dtype=torch.float32)
        invalid_mask = torch.zeros((batch_size, num_elements, num_points), dtype=torch.bool)
        pos_enc = torch.zeros((1, 1, num_points, self.dim_in), dtype=torch.float32)

        output = self.model.forward(inputs, invalid_mask, pos_enc)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, num_elements, self.dim_in))


class TestMultiheadAttentionGlobalHead(unittest.TestCase):
    """Test MultiheadAttentionGlobalHead layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.global_embedding_size = 256
        self.num_timesteps = 12
        self.num_outputs = 3
        self.model = MultiheadAttentionGlobalHead(self.global_embedding_size, self.num_timesteps, self.num_outputs)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        batch_size = 2
        num_elements = 10
        inputs = torch.zeros((num_elements, batch_size, self.global_embedding_size), dtype=torch.float32)
        type_embedding = torch.ones((num_elements, batch_size, self.global_embedding_size), dtype=torch.long)
        invalid_mask = torch.zeros((batch_size, num_elements), dtype=torch.bool)

        output, attns = self.model.forward(inputs, type_embedding, invalid_mask)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, self.num_timesteps, self.num_outputs))

        self.assertIsInstance(attns, torch.Tensor)
        assert attns is not None
        self.assertEqual(attns.shape, (batch_size, 1, num_elements))


class TestUrbanDriverOpenLoop(unittest.TestCase):
    """Test UrbanDriverOpenLoopModel model."""

    def setUp(self) -> None:
        """Set up the test."""
        self.model_params = UrbanDriverOpenLoopModelParams(
            local_embedding_size=256,
            global_embedding_size=256,
            num_subgraph_layers=3,
            global_head_dropout=0.0,
        )

        self.feature_params = UrbanDriverOpenLoopModelFeatureParams(
            feature_types={
                'NONE': -1,
                'EGO': 0,
                'VEHICLE': 1,
                'BICYCLE': 2,
                'PEDESTRIAN': 3,
                'LANE': 4,
                'STOP_LINE': 5,
                'CROSSWALK': 6,
                'LEFT_BOUNDARY': 7,
                'RIGHT_BOUNDARY': 8,
                'ROUTE_LANES': 9,
            },
            total_max_points=20,
            feature_dimension=8,
            agent_features=['VEHICLE', 'BICYCLE', 'PEDESTRIAN'],
            ego_dimension=3,
            agent_dimension=8,
            max_agents=30,
            past_trajectory_sampling=TrajectorySampling(time_horizon=2.0, num_poses=4),
            map_features=['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES'],
            max_elements={
                'LANE': 30,
                'LEFT_BOUNDARY': 30,
                'RIGHT_BOUNDARY': 30,
                'STOP_LINE': 20,
                'CROSSWALK': 20,
                'ROUTE_LANES': 30,
            },
            max_points={
                'LANE': 20,
                'LEFT_BOUNDARY': 20,
                'RIGHT_BOUNDARY': 20,
                'STOP_LINE': 20,
                'CROSSWALK': 20,
                'ROUTE_LANES': 20,
            },
            vector_set_map_feature_radius=35,
            interpolation_method='linear',
            disable_map=False,
            disable_agents=False,
        )

        self.target_params = UrbanDriverOpenLoopModelTargetParams(
            num_output_features=36,
            future_trajectory_sampling=TrajectorySampling(time_horizon=6.0, num_poses=12),
        )

    def _build_model(self) -> UrbanDriverOpenLoopModel:
        """
        Creates a new instance of a UrbanDriverOpenLoop with some default parameters.
        """
        model = UrbanDriverOpenLoopModel(self.model_params, self.feature_params, self.target_params)

        return model

    def _build_input_features(self, device: torch.device, include_agents: bool) -> FeaturesType:
        """
        Creates a set of input features for use with unit testing.
        :param device: The device on which to create the tensors.
        :param include_agents: If true, the generated input features will have agents.
            If not, then there will be no agents in the agents feature.
        :return: FeaturesType to be consumed by the model
        """
        # Numbers chosen arbitrarily
        num_frames = 5
        num_agents = num_frames if include_agents else 0

        coords: Dict[str, List[torch.Tensor]] = dict()
        traffic_light_data: Dict[str, List[torch.Tensor]] = dict()
        availabilities: Dict[str, List[torch.BoolTensor]] = dict()

        for feature_name in self.feature_params.map_features:
            coords[feature_name] = [
                torch.zeros(
                    (
                        self.feature_params.max_elements[feature_name],
                        self.feature_params.max_points[feature_name],
                        VectorSetMap.coord_dim(),
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            ]
            availabilities[feature_name] = [
                torch.ones(
                    (self.feature_params.max_elements[feature_name], self.feature_params.max_points[feature_name]),
                    dtype=torch.bool,
                    device=device,
                )
            ]

        # tl status for lanes
        traffic_light_data['LANE'] = [
            torch.zeros(
                (
                    self.feature_params.max_elements['LANE'],
                    self.feature_params.max_points['LANE'],
                    VectorSetMap.traffic_light_status_dim(),
                ),
                dtype=torch.float32,
                device=device,
            )
        ]

        vector_set_map_feature = VectorSetMap(
            coords=coords,
            traffic_light_data=traffic_light_data,
            availabilities=availabilities,
        )

        ego_agents = [torch.zeros((num_frames, GenericAgents.ego_state_dim()), dtype=torch.float32, device=device)]
        agent_agents = {
            feature_name: [
                torch.zeros(
                    (num_frames, num_agents, GenericAgents.agents_states_dim()), dtype=torch.float32, device=device
                )
            ]
            for feature_name in self.feature_params.agent_features
        }

        generic_agents_feature = GenericAgents(ego=ego_agents, agents=agent_agents)

        return {
            "vector_set_map": vector_set_map_feature,
            "generic_agents": generic_agents_feature,
        }

    def _find_free_port(self) -> int:
        # TODO: move to shared utils across model tests
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
        # TODO: move to shared utils across model tests
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
        Tests that the UrbanDriverOpenLoop model can train with DDP.
        This test was developed in response to an error related to zero agent input
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
                input_features = self._build_input_features(device, include_agents=include_agents)
                predictions = ddp_model.forward(input_features)
                self._assert_valid_output(predictions)
                self._perform_backprop_step(optimizer, loss_function, predictions)


if __name__ == "__main__":
    unittest.main()
