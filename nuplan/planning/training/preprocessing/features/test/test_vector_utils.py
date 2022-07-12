import unittest

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from nuplan.planning.training.preprocessing.features.vector_utils import (
    rotate_coords,
    scale_coords,
    translate_coords,
    xflip_coords,
    yflip_coords,
)


class TestVectorUtils(unittest.TestCase):
    """Test vector-based feature utility functions."""

    def setUp(self) -> None:
        """Set up test case."""
        self.coords: npt.NDArray[np.float64] = np.array(
            [
                [[0.0, 0.0], [-1.0, 1.0], [1.0, 1.0]],
                [[1.0, 0.0], [-1.0, -1.0], [1.0, -1.0]],
            ]
        )

    def test_rotate_coords(self) -> None:
        """
        Test vector feature coordinate rotation.
        """
        quaternion = Quaternion(axis=[1, 0, 0], angle=3.14159265)

        expected_result: npt.NDArray[np.float64] = np.array(
            [
                [[0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]],
                [[1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]],
            ]
        )

        result = rotate_coords(self.coords, quaternion)

        np.testing.assert_allclose(expected_result, result)

    def test_translate_coords(self) -> None:
        """
        Test vector feature coordinate translation.
        """
        translation_value: npt.NDArray[np.float64] = np.array([1.0, 0.0, -1.0])

        expected_result: npt.NDArray[np.float64] = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]],
                [[2.0, 0.0], [0.0, -1.0], [2.0, -1.0]],
            ]
        )

        result = translate_coords(self.coords, translation_value)

        np.testing.assert_allclose(expected_result, result)

    def test_scale_coords(self) -> None:
        """
        Test vector feature coordinate scaling.
        """
        scale_value: npt.NDArray[np.float64] = np.array([-2.0, 0.0, -1.0])

        expected_result: npt.NDArray[np.float64] = np.array(
            [
                [[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
                [[-2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
            ]
        )

        result = scale_coords(self.coords, scale_value)

        np.testing.assert_allclose(expected_result, result)

    def test_xflip_coords(self) -> None:
        """
        Test flipping vector feature coordinates about X-axis.
        """
        expected_result: npt.NDArray[np.float64] = np.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]],
                [[-1.0, 0.0], [1.0, -1.0], [-1.0, -1.0]],
            ]
        )

        result = xflip_coords(self.coords)

        np.testing.assert_allclose(expected_result, result)

    def test_yflip_coords(self) -> None:
        """
        Test flipping vector feature coordinates about Y-axis.
        """
        expected_result: npt.NDArray[np.float64] = np.array(
            [
                [[0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]],
                [[1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]],
            ]
        )

        result = yflip_coords(self.coords)

        np.testing.assert_allclose(expected_result, result)


if __name__ == '__main__':
    unittest.main()
