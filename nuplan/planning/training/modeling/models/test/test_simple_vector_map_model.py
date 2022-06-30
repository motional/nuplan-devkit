import unittest
from typing import Dict, List

import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.simple_vector_map_model import VectorMapSimpleMLP


class TestVectorMapSimpleMLP(unittest.TestCase):
    """Test graph attention layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.num_output_features = 36
        self.hidden_size = 128
        self.vector_map_feature_radius = 20
        self.past_trajectory_sampling = TrajectorySampling(num_poses=4, time_horizon=1.5)
        self.future_trajectory_sampling = TrajectorySampling(num_poses=12, time_horizon=6)
        self.model = VectorMapSimpleMLP(
            num_output_features=self.num_output_features,
            hidden_size=self.hidden_size,
            vector_map_feature_radius=self.vector_map_feature_radius,
            past_trajectory_sampling=self.past_trajectory_sampling,
            future_trajectory_sampling=self.future_trajectory_sampling,
        )

    def test_scripts_properly(self) -> None:
        """
        Test that the VectorMapSimpleMLP model scripts properly.
        """
        dummy_tensor_input: Dict[str, torch.Tensor] = {}
        dummy_list_tensor_input = {
            "vector_map.coords": [torch.zeros(4808, 2, 2), torch.zeros(4808, 2, 2)],
            "agents.ego": [
                torch.zeros(5, 3),
                torch.zeros(5, 3),
            ],
            "agents.agents": [torch.zeros(5, 54, 8), torch.zeros(5, 54, 8)],
        }
        dummy_list_list_tensor_input: Dict[str, List[List[torch.Tensor]]] = {}

        scripted_module = torch.jit.script(self.model)

        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_module.scriptable_forward(
            dummy_tensor_input, dummy_list_tensor_input, dummy_list_list_tensor_input
        )

        py_tensors, py_list_tensors, py_list_list_tensors = scripted_module.scriptable_forward(
            dummy_tensor_input, dummy_list_tensor_input, dummy_list_list_tensor_input
        )

        self.assertEqual(1, len(scripted_tensors))
        self.assertEqual(0, len(scripted_list_tensors))
        self.assertEqual(0, len(scripted_list_list_tensors))

        self.assertEqual(1, len(py_tensors))
        self.assertEqual(0, len(py_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))

        torch.testing.assert_allclose(py_tensors["trajectory"], scripted_tensors["trajectory"])


if __name__ == "__main__":
    unittest.main()
