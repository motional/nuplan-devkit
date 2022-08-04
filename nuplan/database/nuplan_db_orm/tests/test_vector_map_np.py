import unittest
from typing import Dict
from unittest.mock import Mock, patch

import numpy as np
import numpy.typing as npt

from nuplan.database.nuplan_db_orm.vector_map_np import VectorMapNp


class TestVectorMapNp(unittest.TestCase):
    """
    Tests the VectorMapNp class
    """

    def setUp(self) -> None:
        """
        Sets up for the test cases
        """
        coords: npt.NDArray[np.float64] = np.ones([1, 2, 2], dtype=np.float32)
        multi_scale_connections = Dict[int, npt.NDArray[np.float64]]
        self.vector_map_np = VectorMapNp(coords, multi_scale_connections)

    def test_translate(self) -> None:
        """
        Tests the translate method
        """
        # Setup
        vector_map_np = self.vector_map_np
        translate = [1.0, 1.0, 0.0]
        expected_coords = 2.0 * np.ones([1, 2, 2], dtype=np.float32)

        # Call the method under test
        result = vector_map_np.translate(translate)

        # Assertions
        self.assertTrue(np.array_equal(result.coords, expected_coords))

    @patch('nuplan.database.nuplan_db_orm.vector_map_np.np.dot', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.vector_map_np.np.concatenate', autospec=True)
    def test_rotate(self, concatenate_mock: Mock, dot_mock: Mock) -> None:
        """
        Tests the rotate method
        """
        # Setup
        vector_map_np = self.vector_map_np
        quarternion = Mock()

        # Call the method under test
        vector_map_np.rotate(quarternion)

        # Assertions
        dot_mock.assert_called_once()
        concatenate_mock.assert_called_once()

    def test_scale(self) -> None:
        """
        Tests the scale method
        """
        # Setup
        vector_map_np = self.vector_map_np
        scale = [3.0, 3.0, 3.0]
        expected_coords = 3.0 * np.ones([1, 2, 2], dtype=np.float32)

        # Call the method under test
        result = vector_map_np.scale(scale)

        # Assertions
        self.assertTrue(np.array_equal(result.coords, expected_coords))

    def test_xflip(self) -> None:
        """
        Tests the xflip method
        """
        # Setup
        vector_map_np = self.vector_map_np
        expected_coords: npt.NDArray[np.float64] = np.array([[[-1, 1], [-1, 1]]])

        # Call the method under test
        result = vector_map_np.xflip()

        # Assertions
        self.assertTrue(np.array_equal(result.coords, expected_coords))

    def test_yflip(self) -> None:
        """
        Tests the yflip method
        """
        # Setup
        vector_map_np = self.vector_map_np
        expected_coords: npt.NDArray[np.float64] = np.array([[[1, -1], [1, -1]]])

        # Call the method under test
        result = vector_map_np.yflip()

        # Assertions
        self.assertTrue(np.array_equal(result.coords, expected_coords))


if __name__ == "__main__":
    unittest.main()
