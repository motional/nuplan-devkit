import random
import unittest
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import numpy.typing as npt

from nuplan.database.tests.test_utils_nuplan_db import get_test_nuplan_db, get_test_nuplan_egopose


class TestEgoPose(unittest.TestCase):
    """Tests the EgoPose class"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.ego_pose = get_test_nuplan_egopose()

    @patch('nuplan.database.nuplan_db_orm.ego_pose.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the _session property"""
        # Setup
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock

        # Call the method under test
        result = self.ego_pose._session

        # Assertions
        inspect_mock.assert_called_once_with(self.ego_pose)
        self.assertEqual(result, session_mock)

    @patch('nuplan.database.nuplan_db_orm.ego_pose.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the __repr__ method"""
        # Call the method under test
        result = self.ego_pose.__repr__()

        # Assertions
        simple_repr_mock.assert_called_once_with(self.ego_pose)
        self.assertEqual(result, simple_repr_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.ego_pose.Quaternion', autospec=True)
    def test_quaternion(self, quaternion_mock: Mock) -> None:
        """Tests the quaternion method"""
        # Call the method under test
        result = self.ego_pose.quaternion

        # Assertions
        quaternion_mock.assert_called_once_with(self.ego_pose.qw, self.ego_pose.qx, self.ego_pose.qy, self.ego_pose.qz)
        self.assertEqual(result, quaternion_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.ego_pose.np.array', autospec=True)
    def test_translation_np(self, np_array_mock: Mock) -> None:
        """Tests the translation_np method"""
        # Call the method under test
        result = self.ego_pose.translation_np

        # Assertions
        np_array_mock.assert_called_with(
            [
                self.ego_pose.x,
                self.ego_pose.y,
                self.ego_pose.z,
            ]
        )
        self.assertEqual(result, np_array_mock.return_value)

    def test_trans_matrix_and_inv(self) -> None:
        """Tests the transformation matrix and it's inverse method"""
        # Call the methods under test
        trans_matrix = self.ego_pose.trans_matrix
        trans_matrix_inv = self.ego_pose.trans_matrix_inv

        # Result of multiplying a matrix with its inverse,
        # should result in a 4x4 identity matrix
        result = np.matmul(trans_matrix, trans_matrix_inv)

        # Assertions
        np.testing.assert_allclose(result, np.identity(4), atol=1e-3)

    def test_rotate_2d_points2d_to_ego_vehicle_frame(self) -> None:
        """Tests the rotate_2d_points2d_to_ego_vehicle_frame method"""
        # Setup
        points2d: npt.NDArray[np.float32] = np.ones([1, 2], dtype=np.float32)

        # Call the method under test
        result = self.ego_pose.rotate_2d_points2d_to_ego_vehicle_frame(points2d)

        # Assertions
        self.assertEqual(result.ndim, 2)

    def test_get_map_crop_dimensions(self) -> None:
        """
        Test that map crop method produces map of the correct dimensions.
        Test time: 10.569s
        """
        # Setup
        xrange = (-60, 60)
        yrange = (-60, 60)
        rotate_face_up = False
        map_layer_description = 'intensity'
        map_layer_precision = 0.1
        map_scale = 1 / map_layer_precision
        num_samples = 10
        db = get_test_nuplan_db()
        selected_indices = random.sample(list(range(len(db.ego_pose))), num_samples)

        expected_dimensions = (
            (xrange[1] - xrange[0]) * map_scale,
            (yrange[1] - yrange[0]) * map_scale,
        )
        ego_pose_list = db.ego_pose

        for i in selected_indices:
            current_ego_pose = ego_pose_list[i]
            if current_ego_pose.lidar_pc is None:
                continue

            # Call the method under test
            map_crop = current_ego_pose.get_map_crop(
                maps_db=db.maps_db,
                xrange=xrange,
                yrange=yrange,
                map_layer_name=map_layer_description,
                rotate_face_up=rotate_face_up,
            )

            # Assertions
            self.assertTrue(map_crop[0] is not None)
            self.assertEqual(expected_dimensions, map_crop[0].shape, f"Dimensions failed at ego pose index {i}")

    def test_get_vector_map(self) -> None:
        """Tests the get vector map method"""
        # Setup
        xrange = (-60, 60)
        yrange = (-60, 60)
        db = get_test_nuplan_db()
        num_samples = 10
        selected_indices = random.sample(list(range(len(db.ego_pose))), num_samples)
        ego_pose_list = db.ego_pose

        for i in selected_indices:
            current_ego_pose = ego_pose_list[i]
            if current_ego_pose.lidar_pc is None:
                continue

            # Call the method under test
            result = current_ego_pose.get_vector_map(db.maps_db, xrange, yrange)

            # Assertions
            # Check if the result is not None
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
