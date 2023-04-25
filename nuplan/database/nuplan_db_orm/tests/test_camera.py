import unittest
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

from nuplan.database.tests.test_utils_nuplan_db import get_test_nuplan_camera


class TestCamera(unittest.TestCase):
    """Test class Camera"""

    def setUp(self) -> None:
        """
        Initializes a test Camera
        """
        self.camera = get_test_nuplan_camera()

    @patch("nuplan.database.nuplan_db_orm.camera.inspect", autospec=True)
    def test_session(self, inspect: Mock) -> None:
        """
        Tests _session method
        """
        # Setup
        mock_session = PropertyMock()
        inspect.return_value = Mock()
        inspect.return_value.session = mock_session

        # Call method under test
        result = self.camera._session()

        # Assertions
        inspect.assert_called_once_with(self.camera)
        mock_session.assert_called_once()
        self.assertEqual(result, mock_session.return_value)

    @patch("nuplan.database.nuplan_db_orm.camera.simple_repr", autospec=True)
    def test_repr(self, simple_repr: Mock) -> None:
        """
        Tests string representation
        """
        # Call method under test
        result = self.camera.__repr__()

        # Assertions
        simple_repr.assert_called_once_with(self.camera)
        self.assertEqual(result, simple_repr.return_value)

    @patch("nuplan.database.nuplan_db_orm.camera.np.array", autospec=True)
    def test_intrinsic_np(self, np_array: Mock) -> None:
        """
        Test property - camera intrinsic.
        """
        # Call method under test
        result = self.camera.intrinsic_np

        # Assertions
        np_array.assert_called_once_with(self.camera.intrinsic)
        self.assertEqual(result, np_array.return_value)

    @patch("nuplan.database.nuplan_db_orm.camera.np.array", autospec=True)
    def test_distortion_np(self, np_array: Mock) -> None:
        """
        Test property - camera distrotion.
        """
        # Call method under test
        result = self.camera.distortion_np

        # Assertions
        np_array.assert_called_once_with(self.camera.distortion)
        self.assertEqual(result, np_array.return_value)

    @patch("nuplan.database.nuplan_db_orm.camera.np.array", autospec=True)
    def test_translation_np(self, np_array: Mock) -> None:
        """
        Test property - translation.
        """
        # Call method under test
        result = self.camera.translation_np

        # Assertions
        np_array.assert_called_once_with(self.camera.translation)
        self.assertEqual(result, np_array.return_value)

    def test_quaternion(self) -> None:
        """
        Test property - rotation in quaternion.
        """
        # Call method under test
        result = self.camera.quaternion

        # Assertions
        np.testing.assert_array_equal(self.camera.rotation, result.elements)

    def test_trans_matrix_and_inv(self) -> None:
        """
        Test two properties - transformation matrix and its inverse.
        """
        # Call method under test
        trans_mat = self.camera.trans_matrix
        inv_trans_mat = self.camera.trans_matrix_inv

        # Assertions
        np.testing.assert_allclose(trans_mat @ inv_trans_mat, np.eye(4), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
