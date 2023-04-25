import unittest
from unittest.mock import Mock, patch

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.transform import (
    rotate,
    rotate_2d,
    rotate_angle,
    transform,
    translate,
    translate_laterally,
    translate_longitudinally,
    translate_longitudinally_and_laterally,
)


class TestTransform(unittest.TestCase):
    """Tests for transform functions"""

    def test_rotate_2d(self) -> None:
        """Tests rotation of 2D point"""
        # Setup
        point = Point2D(1, 0)
        rotation_matrix: npt.NDArray[np.float32] = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        # Function call
        result = rotate_2d(point, rotation_matrix)

        # Checks
        self.assertEqual(result, Point2D(0, 1))

    def test_translate(self) -> None:
        """Tests translate"""
        # Setup
        pose = StateSE2(3, 5, np.pi / 4)
        translation: npt.NDArray[np.float32] = np.array([1, 2], dtype=np.float32)

        # Function call
        result = translate(pose, translation)

        # Checks
        self.assertEqual(result, StateSE2(4, 7, np.pi / 4))

    def test_rotate(self) -> None:
        """Tests rotation of SE2 pose by rotation matrix"""
        # Setup
        pose = StateSE2(1, 2, np.pi / 4)
        rotation_matrix: npt.NDArray[np.float32] = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        # Function call
        result = rotate(pose, rotation_matrix)

        # Checks
        self.assertAlmostEqual(result.x, -2)
        self.assertAlmostEqual(result.y, 1)
        self.assertAlmostEqual(result.heading, -np.pi / 4)

    def test_rotate_angle(self) -> None:
        """Tests rotation of SE2 pose by angle (in radian)"""
        # Setup
        pose = StateSE2(1, 2, np.pi / 4)
        angle = -np.pi / 2

        # Function call
        result = rotate_angle(pose, angle)

        # Checks
        self.assertAlmostEqual(result.x, -2)
        self.assertAlmostEqual(result.y, 1)
        self.assertAlmostEqual(result.heading, -np.pi / 4)

    def test_transform(self) -> None:
        """Tests transformation of SE2 pose"""
        # Setup
        pose = StateSE2(1, 2, 0)
        transform_matrix: npt.NDArray[np.float32] = np.array([[-3, -2, 5], [0, -1, 4], [0, 0, 1]], dtype=np.float32)

        # Function call
        result = transform(pose, transform_matrix)

        # Checks
        self.assertAlmostEqual(result.x, 2)
        self.assertAlmostEqual(result.y, 0)
        self.assertAlmostEqual(result.heading, np.pi, places=4)

    @patch("nuplan.common.geometry.transform.translate")
    def test_translate_longitudinally(self, mock_translate: Mock) -> None:
        """Tests longitudinal translation"""
        # Setup
        pose = StateSE2(1, 2, np.arctan(1 / 3))

        # Function call
        result = translate_longitudinally(pose, np.sqrt(10))

        # Checks
        np.testing.assert_array_almost_equal(mock_translate.call_args.args[1], np.array([3, 1]))
        self.assertEqual(result, mock_translate.return_value)

    @patch("nuplan.common.geometry.transform.translate")
    def test_translate_laterally(self, mock_translate: Mock) -> None:
        """Tests lateral translation"""
        # Setup
        pose = StateSE2(1, 2, np.arctan(1 / 3))

        # Function call
        result = translate_laterally(pose, np.sqrt(10))

        # Checks
        np.testing.assert_array_almost_equal(mock_translate.call_args.args[1], np.array([-1, 3]))
        self.assertEqual(result, mock_translate.return_value)

    @patch("nuplan.common.geometry.transform.translate")
    def test_translate_longitudinally_and_laterally(self, mock_translate: Mock) -> None:
        """Tests longitudinal and lateral translation"""
        # Setup
        pose = StateSE2(1, 2, np.arctan(1 / 3))

        # Function call
        result = translate_longitudinally_and_laterally(pose, np.sqrt(10), np.sqrt(10))

        # Checks
        np.testing.assert_array_almost_equal(mock_translate.call_args.args[1], np.array([2, 4]))
        self.assertEqual(result, mock_translate.return_value)


if __name__ == "__main__":
    unittest.main()
