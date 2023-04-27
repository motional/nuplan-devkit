import unittest
from unittest.mock import Mock, patch

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import (
    absolute_to_relative_poses,
    matrix_from_pose,
    numpy_array_to_absolute_pose,
    numpy_array_to_absolute_velocity,
    pose_from_matrix,
    relative_to_absolute_poses,
    vector_2d_from_magnitude_angle,
)


class TestConvert(unittest.TestCase):
    """Tests for convert functions"""

    def test_pose_from_matrix(self) -> None:
        """Tests conversion from 3x3 transformation matrix to a 2D pose"""
        # Setup
        transform_matrix: npt.NDArray[np.float32] = np.array(
            [[np.sqrt(3) / 2, -0.5, 2], [0.5, np.sqrt(3) / 2, 2], [0, 0, 1]], dtype=np.float32
        )
        expected_pose = StateSE2(2, 2, np.pi / 6)

        # Function call
        result = pose_from_matrix(transform_matrix=transform_matrix)

        # Assertions
        self.assertAlmostEqual(result.x, expected_pose.x)
        self.assertAlmostEqual(result.y, expected_pose.y)
        self.assertAlmostEqual(result.heading, expected_pose.heading)

        # Should raise if the transform dimensions are incorrect
        with self.assertRaises(RuntimeError):
            bad_matrix: npt.NDArray[np.float32] = np.array(
                [[np.sqrt(3) / 2, -0.5, 2], [0.5, np.sqrt(3) / 2, 2]], dtype=np.float32
            )

            _ = pose_from_matrix(transform_matrix=bad_matrix)

    def test_matrix_from_pose(self) -> None:
        """Tests conversion from 2D pose to a 3x3 transformation matrix"""
        # Setup
        pose = StateSE2(2, 2, np.pi / 6)
        expected_transform_matrix: npt.NDArray[np.float32] = np.array(
            [[np.sqrt(3) / 2, -0.5, 2], [0.5, np.sqrt(3) / 2, 2], [0, 0, 1]], dtype=np.float32
        )

        # Function call
        result = matrix_from_pose(pose=pose)

        # Assertions
        np.testing.assert_array_almost_equal(result, expected_transform_matrix)

    def test_absolute_to_relative_poses(self) -> None:
        """Tests conversion of a list of SE2 poses from absolute to relative coordinates"""
        # Setup
        inv_sqrt_2 = 1 / np.sqrt(2)

        origin = StateSE2(1, 1, np.pi / 4)

        poses = [origin, StateSE2(1, 1, np.pi / 2), StateSE2(1, 1, np.pi / 4), StateSE2(2, 3, 0), StateSE2(3, 2, 0)]

        expected_poses = [
            StateSE2(0, 0, 0),
            StateSE2(0, 0, np.pi / 4),
            StateSE2(0, 0, 0),
            StateSE2(3 * inv_sqrt_2, inv_sqrt_2, -np.pi / 4),
            StateSE2(3 * inv_sqrt_2, -inv_sqrt_2, -np.pi / 4),
        ]

        # Function call
        result = absolute_to_relative_poses(poses)

        # Assertions
        for i in range(len(result)):
            self.assertAlmostEqual(result[i].x, expected_poses[i].x)
            self.assertAlmostEqual(result[i].y, expected_poses[i].y)
            self.assertAlmostEqual(result[i].heading, expected_poses[i].heading)

    def test_relative_to_absolute_poses(self) -> None:
        """Tests conversion of a list of SE2 poses from relative to absolute coordinates"""
        # Setup
        inv_sqrt_2 = 1 / np.sqrt(2)

        origin = StateSE2(1, 1, np.pi / 4)

        poses = [
            StateSE2(0, 0, np.pi / 4),
            StateSE2(0, 0, 0),
            StateSE2(3 * inv_sqrt_2, inv_sqrt_2, -np.pi / 4),
            StateSE2(3 * inv_sqrt_2, -inv_sqrt_2, -np.pi / 4),
        ]

        expected_poses = [StateSE2(1, 1, np.pi / 2), StateSE2(1, 1, np.pi / 4), StateSE2(2, 3, 0), StateSE2(3, 2, 0)]

        # Funcsigned lateral distance of ego to polygontion call
        result = relative_to_absolute_poses(origin, poses)

        # Assertions
        for i in range(len(result)):
            self.assertAlmostEqual(result[i].x, expected_poses[i].x)
            self.assertAlmostEqual(result[i].y, expected_poses[i].y)
            self.assertAlmostEqual(result[i].heading, expected_poses[i].heading)

    def test_input_numpy_array_to_absolute_velocity(self) -> None:
        """Tests input validation of numpy_array_to_absolute_velocity"""
        # Setup
        np_velocities = np.random.random(size=(10, 3))

        # Function call
        with self.assertRaises(AssertionError):
            numpy_array_to_absolute_velocity(StateSE2(0, 0, 0), np_velocities)

    @patch("nuplan.common.geometry.convert.relative_to_absolute_poses")
    def test_numpy_array_to_absolute_velocity(self, mock_relative_to_absolute_poses: Mock) -> None:
        """Tests conversion from relative numpy velocities to list of absolute velocities"""
        # Setup
        np_velocities: npt.NDArray[np.float32] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        num_velocities = len(np_velocities)
        mock_relative_to_absolute_poses.side_effect = lambda _, r_s: r_s

        # Function call
        result = numpy_array_to_absolute_velocity("origin", np_velocities)

        # Assertions
        mock_relative_to_absolute_poses.assert_called_once()

        self.assertEqual(num_velocities, len(result))

        for i in range(num_velocities):
            self.assertEqual(result[i].x, np_velocities[i][0])
            self.assertEqual(result[i].y, np_velocities[i][1])

    def test_input_numpy_array_to_absolute_pose_input(self) -> None:
        """Tests input validation of numpy_array_to_absolute_pose_input"""
        # Setup
        np_poses = np.random.random((10, 2))

        # Function call
        with self.assertRaises(AssertionError):
            numpy_array_to_absolute_pose(StateSE2(0, 0, 0), np_poses)

    @patch("nuplan.common.geometry.convert.relative_to_absolute_poses")
    def test_numpy_array_to_absolute_pose(self, mock_relative_to_absolute_poses: Mock) -> None:
        """Tests conversion from relative numpy poses to list of absolute StateSE2 objects."""
        # Setup
        np_poses = np.random.random((10, 3))
        mock_relative_to_absolute_poses.side_effect = lambda _, r_s: r_s

        # Function call
        result = numpy_array_to_absolute_pose("origin", np_poses)

        # Assertions
        mock_relative_to_absolute_poses.assert_called_once()
        for np_p, se2_p in zip(np_poses, result):
            self.assertEqual(np_p[0], se2_p.x)
            self.assertEqual(np_p[1], se2_p.y)
            self.assertEqual(np_p[2], se2_p.heading)

    @patch("nuplan.common.geometry.convert.np")
    @patch("nuplan.common.geometry.convert.StateVector2D")
    def test_vector_2d_from_magnitude_angle(self, vector: Mock, mock_np: Mock) -> None:
        """Tests that projection to vector works as expected."""
        magnitude = Mock()
        angle = Mock()

        result = vector_2d_from_magnitude_angle(magnitude, angle)

        self.assertEqual(result, vector.return_value)
        vector.assert_called_once_with(mock_np.cos() * magnitude, mock_np.sin() * angle)


if __name__ == "__main__":
    unittest.main()
