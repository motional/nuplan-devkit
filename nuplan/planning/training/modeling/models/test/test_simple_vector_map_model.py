import os
import socket
import unittest
from typing import Callable, Dict, List

import torch
import torch.nn
import torch.optim
from torch.nn.parallel import DistributedDataParallel as DDP

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.simple_vector_map_model import VectorMapSimpleMLP
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap


def _create_empty_vector_map_for_test(device: torch.device) -> VectorMap:
    """
    Helper function to create a dummy empty vector map solely for testing purposes.
    :param device: The device on which to place the constructed vector map.
    :return: A VectorMap with batch size 1 but otherwise empty features.
    """
    coords = [torch.zeros(size=(0, 2, 2), dtype=torch.float32, device=device)]
    lane_groupings = [[torch.zeros(size=(0,), dtype=torch.float32, device=device)]]
    multi_scale_connections = {1: [torch.zeros(size=(0, 2), dtype=torch.float32, device=device)]}
    on_route_status = [
        torch.zeros(size=(0, VectorMap.on_route_status_encoding_dim()), dtype=torch.float32, device=device)
    ]
    traffic_light_data = [torch.zeros(size=(0, 4), dtype=torch.float32, device=device)]

    return VectorMap(
        coords=coords,
        lane_groupings=lane_groupings,
        multi_scale_connections=multi_scale_connections,
        on_route_status=on_route_status,
        traffic_light_data=traffic_light_data,
    )


class TestVectorMapSimpleMLP(unittest.TestCase):
    """Test graph attention layer."""

    def _build_model(self) -> VectorMapSimpleMLP:
        """
        Creates a new instance of a VectorMapSimpleMLP with some default parameters.
        """
        num_output_features = 36
        hidden_size = 128
        vector_map_feature_radius = 20
        past_trajectory_sampling = TrajectorySampling(num_poses=4, time_horizon=1.5)
        future_trajectory_sampling = TrajectorySampling(num_poses=12, time_horizon=6)
        model = VectorMapSimpleMLP(
            num_output_features=num_output_features,
            hidden_size=hidden_size,
            vector_map_feature_radius=vector_map_feature_radius,
            past_trajectory_sampling=past_trajectory_sampling,
            future_trajectory_sampling=future_trajectory_sampling,
        )

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
        num_coords = 1000
        num_groupings = 100
        num_multi_scale_connections = 800
        num_agents = num_frames if include_agents else 0

        vector_map_coords = [
            torch.zeros(
                (num_coords, VectorMap.lane_coord_dim(), VectorMap.lane_coord_dim()), dtype=torch.float32, device=device
            )
        ]
        vector_map_lane_groupings = [[torch.zeros((num_groupings), device=device)]]
        multi_scale_connections = {1: [torch.zeros((num_multi_scale_connections, 2), device=device)]}
        on_route_status = [torch.zeros((num_coords, VectorMap.on_route_status_encoding_dim()), device=device)]
        traffic_light_data = [torch.zeros((num_coords, 4), device=device)]

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

    def _assert_valid_output(self, model_output: TargetsType) -> None:
        """
        Validates that the output from the model has the correct keys and that the tensor is of the correct type.
        :param model_output: The output from the model.
        """
        self.assertTrue("trajectory" in model_output)
        self.assertTrue(isinstance(model_output["trajectory"], Trajectory))

        predicted_trajectory: Trajectory = model_output["trajectory"]
        self.assertIsNotNone(predicted_trajectory.data)

        # Additional asserts handled in `__post_init__` of dataclass.
        # No need to add redundant checks here.

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
        :param port: The starting to use for the gloo server. If taken, it will increment by 1 until a free port is found.
        :param max_port: The maximum port number to try.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self._find_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        torch.distributed.init_process_group(backend="gloo")

    def _assert_valid_gradients_for_model(self, model: torch.nn.Module) -> None:
        """
        Validates that trainable parameters in a model have gradients after a backprop operation.
        :param model: The model with parameters to update following a forward/backward pass.
        """
        all_gradients_computed = all(param.grad is not None for param in model.parameters() if param.requires_grad)
        self.assertTrue(all_gradients_computed)

    def test_can_train_distributed(self) -> None:
        """
        Tests that the model can train with DDP.
        This test was developed in response to an error like this one:
        https://discuss.pytorch.org/t/need-help-runtimeerror-expected-to-have-finished-reduction-in-the-prior-iteration-before-starting-a-new-one/119247
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
                self._assert_valid_gradients_for_model(ddp_model)

    def test_scripts_properly(self) -> None:
        """
        Test that the VectorMapSimpleMLP model scripts properly.
        """
        model = self._build_model()
        device = torch.device("cpu")
        input_features = self._build_input_features(device, include_agents=True)

        dummy_tensor_input: Dict[str, torch.Tensor] = {}
        dummy_list_tensor_input = {
            "vector_map.coords": input_features["vector_map"].coords,
            "agents.ego": input_features["agents"].ego,
            "agents.agents": input_features["agents"].agents,
        }
        dummy_list_list_tensor_input: Dict[str, List[List[torch.Tensor]]] = {}

        scripted_module = torch.jit.script(model)

        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_module.scriptable_forward(
            dummy_tensor_input, dummy_list_tensor_input, dummy_list_list_tensor_input
        )

        py_tensors, py_list_tensors, py_list_list_tensors = model.scriptable_forward(
            dummy_tensor_input, dummy_list_tensor_input, dummy_list_list_tensor_input
        )

        self.assertEqual(1, len(scripted_tensors))
        self.assertEqual(0, len(scripted_list_tensors))
        self.assertEqual(0, len(scripted_list_list_tensors))

        self.assertEqual(1, len(py_tensors))
        self.assertEqual(0, len(py_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))

        torch.testing.assert_allclose(py_tensors["trajectory"], scripted_tensors["trajectory"])

    def test_can_train_with_empty_vector_map(self) -> None:
        """In case of zero length vector map features, model training should not crash."""
        device = torch.device("cpu")

        # Overwrite the default test features with an empty VectorMap.
        test_features = self._build_input_features(device=device, include_agents=True)
        test_features["vector_map"] = _create_empty_vector_map_for_test(device=device)

        # Confirm the vector map is invalid.
        self.assertFalse(test_features["vector_map"].is_valid)

        # Setup model and optimizer to test how the empty vector map is processed.
        model = self._build_model().to(device)
        optimizer = torch.optim.RMSprop(model.parameters())
        loss_function = torch.nn.MSELoss()

        # Run a forward and backward pass to test backprop works.
        predictions = model.forward(test_features)
        self._assert_valid_output(predictions)
        self._perform_backprop_step(optimizer, loss_function, predictions)

        self._assert_valid_gradients_for_model(model)


if __name__ == "__main__":
    unittest.main()
