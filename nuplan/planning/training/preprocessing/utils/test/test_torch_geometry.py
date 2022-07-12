import math
import unittest

import torch

from nuplan.planning.training.preprocessing.utils.torch_geometry import (
    coordinates_to_local_frame,
    global_state_se2_tensor_to_local,
    state_se2_tensor_to_transform_matrix,
    transform_matrix_to_state_se2_tensor,
)


class TestTorchGeometry(unittest.TestCase):
    """
    A class for testing the functionality of the torch geometry library.
    """

    def test_transform_matrix_conversion_functionality(self) -> None:
        """
        Test the numerical accuracy of the transform matrix conversion utilities.
        """
        initial_state = torch.tensor([5, 6, math.pi / 2], dtype=torch.float32)

        expected_xform_matrix = torch.tensor([[0, -1, 5], [1, 0, 6], [0, 0, 1]], dtype=torch.float32)

        xform_matrix = state_se2_tensor_to_transform_matrix(initial_state, precision=torch.float32)

        torch.testing.assert_allclose(expected_xform_matrix, xform_matrix)

        reverted = transform_matrix_to_state_se2_tensor(xform_matrix, precision=torch.float32)

        torch.testing.assert_allclose(initial_state, reverted)

    def test_transform_matrix_scriptability(self) -> None:
        """
        Tests that the transform matrix conversion utilities script properly.
        """

        class tmp_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                xform = state_se2_tensor_to_transform_matrix(x)
                result = transform_matrix_to_state_se2_tensor(xform)
                return result

        to_script = tmp_module()
        scripted = torch.jit.script(to_script)

        test_input = torch.tensor([1, 2, 3], dtype=torch.float32)

        py_result = to_script.forward(test_input)
        script_result = scripted.forward(test_input)

        torch.testing.assert_allclose(py_result, script_result)

    def test_global_state_se2_tensor_to_local_functionality(self) -> None:
        """
        Tests the numerical accuracy of global_state_se2_tensor_to_local.
        """
        """
           o = coordinates to transform, facing direction >
           # = local reference frame
           y_world
            ^              x_local
            |               ^
            |               |
            |   y_local <---#
            |
            |
            |
         o  |  o>
         V  |
            |
            *---------------------> x_world
               ^
        <o     o

        """
        global_states = torch.tensor(
            [[1, 1, 0], [1, -1, math.pi / 2], [-1, -1, math.pi], [-1, 1, -math.pi / 2]], dtype=torch.float32
        )

        local_state = torch.tensor([5, 5, math.pi / 2], dtype=torch.float32)

        expected_transformed_states = torch.tensor(
            [[-4, 4, -math.pi / 2], [-6, 4, 0], [-6, 6, math.pi / 2], [-4, 6, math.pi]], dtype=torch.float32
        )

        actual_transformed_states = global_state_se2_tensor_to_local(
            global_states, local_state, precision=torch.float32
        )

        torch.testing.assert_allclose(expected_transformed_states, actual_transformed_states)

    def test_global_state_se2_tensor_to_local_scriptability(self) -> None:
        """
        Tests that global_state_se2_tensor_to_local scripts properly.
        """

        class tmp_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, states: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
                result = global_state_se2_tensor_to_local(states, pose)
                return result

        to_script = tmp_module()
        scripted = torch.jit.script(to_script)

        test_states = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        test_pose = torch.tensor([1, 2, 3], dtype=torch.float32)

        py_result = to_script.forward(test_states, test_pose)
        script_result = scripted.forward(test_states, test_pose)

        torch.testing.assert_allclose(py_result, script_result)

    def test_coordinates_to_local_frame_functionality(self) -> None:
        """
        Tests the numerical accuracy of coordinates_to_local_frame.
        """
        """
           o = coordinates to transform
           # = local reference frame
           y_world
            ^              x_local
            |               ^
            |               |
            |   y_local <---#
            |
            |
            |
         o  |  o
            |
            |
            *---------------------> x_world

         o     o

        """
        coordinates = torch.tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=torch.float32)

        local_state = torch.tensor([5, 5, math.pi / 2], dtype=torch.float32)

        expected_coordinates = torch.tensor([[-4, 4], [-6, 4], [-6, 6], [-4, 6]], dtype=torch.float32)

        actual_coordinates = coordinates_to_local_frame(coordinates, local_state, precision=torch.float32)

        torch.testing.assert_allclose(expected_coordinates, actual_coordinates)

    def test_coordinates_to_local_frame_scriptability(self) -> None:
        """
        Tests that the function coordinates_to_local_frame scripts properly.
        """

        class tmp_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, states: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
                result = coordinates_to_local_frame(states, pose, precision=torch.float32)
                return result

        to_script = tmp_module()
        scripted = torch.jit.script(to_script)

        test_states = torch.tensor([[1, 2], [4, 5]], dtype=torch.float32)
        test_pose = torch.tensor([1, 2, 3], dtype=torch.float32)

        py_result = to_script.forward(test_states, test_pose)
        script_result = scripted.forward(test_states, test_pose)

        torch.testing.assert_allclose(py_result, script_result)


if __name__ == "__main__":
    unittest.main()
