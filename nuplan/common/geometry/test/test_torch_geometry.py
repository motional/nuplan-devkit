import math
import unittest

import torch

from nuplan.common.geometry.torch_geometry import (
    coordinates_to_local_frame,
    global_state_se2_tensor_to_local,
    state_se2_tensor_to_transform_matrix,
    state_se2_tensor_to_transform_matrix_batch,
    transform_matrix_to_state_se2_tensor,
    transform_matrix_to_state_se2_tensor_batch,
    vector_set_coordinates_to_local_frame,
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

    def test_transform_matrix_batch_conversion_functionality(self) -> None:
        """
        Test the numerical accuracy of the transform matrix conversion utilities.
        """
        initial_state = torch.tensor([[5, 6, math.pi / 2]], dtype=torch.float32)

        expected_xform_matrix = torch.tensor([[[0, -1, 5], [1, 0, 6], [0, 0, 1]]], dtype=torch.float32)

        xform_matrix = state_se2_tensor_to_transform_matrix_batch(initial_state, precision=torch.float32)

        torch.testing.assert_allclose(expected_xform_matrix, xform_matrix)

        reverted = transform_matrix_to_state_se2_tensor_batch(xform_matrix)

        torch.testing.assert_allclose(initial_state, reverted)

        with self.assertRaises(ValueError):
            # We only accept Nx3x3, not 3x3 tensors.
            misshaped_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
            state_se2_tensor_to_transform_matrix_batch(misshaped_tensor)

    def test_transform_matrix_batch_scriptability(self) -> None:
        """
        Tests that the transform matrix conversion utilities script properly.
        """

        class tmp_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                xform = state_se2_tensor_to_transform_matrix_batch(x)
                result = transform_matrix_to_state_se2_tensor_batch(xform)
                return result

        to_script = tmp_module()
        scripted = torch.jit.script(to_script)

        test_input = torch.tensor([[1, 2, 3]], dtype=torch.float32)

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

    def test_vector_set_coordinates_to_local_frame_functionality(self) -> None:
        """
        Test converting vector set map coordinates from global to local ego frame.
        """
        coords = torch.tensor([[[1, 1], [3, 1], [5, 1]]], dtype=torch.float64)
        avails = torch.ones(coords.shape[:2], dtype=torch.bool)
        anchor_state = torch.tensor([0, 0, 0], dtype=torch.float64)

        result_coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)

        self.assertIsInstance(result_coords, torch.FloatTensor)
        self.assertEqual(result_coords.shape, coords.shape)
        torch.testing.assert_allclose(coords.float(), result_coords)

        # test unexpected shape
        with self.assertRaises(ValueError):
            vector_set_coordinates_to_local_frame(coords[0], avails[0], anchor_state)

        # test mismatching shape
        with self.assertRaises(ValueError):
            vector_set_coordinates_to_local_frame(coords, avails[0], anchor_state)

    def test_vector_set_coordinates_to_local_frame_scriptability(self) -> None:
        """
        Tests that the function vector_set_coordinates_to_local_frame scripts properly.
        """

        class tmp_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, coords: torch.Tensor, avails: torch.Tensor, anchor_state: torch.Tensor) -> torch.Tensor:
                result = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)
                return result

        to_script = tmp_module()
        scripted = torch.jit.script(to_script)

        test_coords = torch.tensor([[[1, 1], [3, 1], [5, 1]]], dtype=torch.float64)
        test_avails = torch.ones(test_coords.shape[:2], dtype=torch.bool)
        test_anchor_state = torch.tensor([0, 0, 0], dtype=torch.float64)

        py_result = to_script.forward(test_coords, test_avails, test_anchor_state)
        script_result = scripted.forward(test_coords, test_avails, test_anchor_state)

        torch.testing.assert_allclose(py_result, script_result)


if __name__ == "__main__":
    unittest.main()
