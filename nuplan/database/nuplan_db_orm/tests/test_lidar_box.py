import math
import unittest
from copy import deepcopy
from sys import maxsize
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
from pyquaternion import Quaternion

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.database.nuplan_db_orm.category import Category
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.log import Log
from nuplan.database.tests.test_utils_nuplan_db import get_test_nuplan_lidar_box, get_test_nuplan_lidar_box_vehicle
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.geometry import quaternion_yaw


class TestLidarBox(unittest.TestCase):
    """Tests the LidarBox class"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.lidar_box_vehicle = get_test_nuplan_lidar_box_vehicle()
        self.lidar_box = get_test_nuplan_lidar_box()

    @patch('nuplan.database.nuplan_db_orm.lidar_box.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the _session property"""
        # Setup
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock

        # Call the method under test
        result = self.lidar_box._session()

        # Assertions
        inspect_mock.assert_called_once_with(self.lidar_box)
        self.assertEqual(result, session_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the __repr__ method"""
        # Call the method under test
        result = self.lidar_box.__repr__()

        # Assertions
        simple_repr_mock.assert_called_once_with(self.lidar_box)
        self.assertEqual(result, simple_repr_mock.return_value)

    def test_log(self) -> None:
        """Tests the log property"""
        # Call the method under test
        result = self.lidar_box.log

        # Assertions
        self.assertIsInstance(result, Log)

    def test_category(self) -> None:
        """Tests the category property"""
        # Call the method under test
        result = self.lidar_box.category

        # Assertions
        self.assertIsInstance(result, Category)

    def test_timestamp(self) -> None:
        """Tests the timestamp property"""
        # Call the method under test
        result = self.lidar_box.timestamp

        # Assertions
        self.assertIsInstance(result, int)

    def test_distance_to_ego(self) -> None:
        """Tests the distance_to_ego property"""
        # Setup
        x = self.lidar_box.x
        y = self.lidar_box.y
        x_ego = self.lidar_box.lidar_pc.ego_pose.x
        y_ego = self.lidar_box.lidar_pc.ego_pose.y

        expected_result = math.sqrt(((x - x_ego) * (x - x_ego)) + ((y - y_ego) * (y - y_ego)))

        # Call the method under test
        actual_result = self.lidar_box.distance_to_ego

        # Assertions
        self.assertEqual(expected_result, actual_result)

    def test_size(self) -> None:
        """Tests the size property"""
        # Call the method under test
        result = self.lidar_box.size

        # Assertions
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], self.lidar_box.width)
        self.assertEqual(result[1], self.lidar_box.length)
        self.assertEqual(result[2], self.lidar_box.height)

    def test_translation(self) -> None:
        """Tests the translation property"""
        # Call the method under test
        result = self.lidar_box.translation

        # Assertions
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], self.lidar_box.x)
        self.assertEqual(result[1], self.lidar_box.y)
        self.assertEqual(result[2], self.lidar_box.z)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.Quaternion', autospec=True)
    def test_rotation(self, quaternion_mock: Mock) -> None:
        """Tests the rotation property"""
        # Call the method under test
        result = self.lidar_box.rotation

        # Assertions
        self.assertIsInstance(result, list)
        quaternion_mock.assert_called()

    @patch('nuplan.database.nuplan_db_orm.lidar_box.Quaternion', autospec=True)
    def test_quaternion(self, quaternion_mock: Mock) -> None:
        """Tests the quaternion property"""
        # Call the method under test
        result = self.lidar_box.quaternion

        # Assertions
        self.assertEqual(result, quaternion_mock.return_value)
        quaternion_mock.assert_called()

    @patch('nuplan.database.nuplan_db_orm.lidar_box.np.array', autospec=True)
    def test_translation_np(self, np_array_mock: Mock) -> None:
        """Tests the translation_np property"""
        # Call the method under test
        result = self.lidar_box.translation_np

        # Assertions
        np_array_mock.assert_called_once_with(self.lidar_box.translation)
        self.assertEqual(result, np_array_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.np.array', autospec=True)
    def test_size_np(self, np_array_mock: Mock) -> None:
        """Tests the size_np property"""
        # Call the method under test
        result = self.lidar_box.size_np

        # Assertions
        np_array_mock.assert_called_once_with(self.lidar_box.size)
        self.assertEqual(result, np_array_mock.return_value)

    def test_get_box_items(self) -> None:
        """Tests the _get_box_items method"""
        # Call the method under test
        result = self.lidar_box._get_box_items()

        # Assertions
        # Returned tuple should contain 2 lists
        self.assertEqual(len(result), 2)

    def test_find_box_out_of_bounds(self) -> None:
        """Tests the _find_box method index is out of bounds"""
        # Call the method under test
        # maxsize is the maximum possible size of the container
        result = self.lidar_box._find_box(maxsize)

        # Assertions
        self.assertEqual(result, None)

    def test_find_box_within_bounds(self) -> None:
        """Tests the _find_box method index is within bounds"""
        # Call the method under test
        result = self.lidar_box._find_box(0)

        # Assertions
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_poses"""
        # Setup
        number, mode, direction = 1, 'n_poses', 'prev'

        # Call the method under test
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)

        # Assertions
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_seconds"""
        # Setup
        number, mode, direction = 1, 'n_seconds', 'prev'

        # Call the method under test
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)

        # Assertions
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev and mode is unknown"""
        # Setup
        number, mode, direction = 1, 'unknown_mode', 'prev'

        # Call the method under test
        with self.assertRaises(ValueError):
            # Should result in a ValueError being raised
            self.lidar_box.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_next_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_poses"""
        # Setup
        number, mode, direction = 1, 'n_poses', 'next'

        # Call the method under test
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)

        # Assertions
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_seconds"""
        # Setup
        number, mode, direction = 1, 'n_seconds', 'next'

        # Call the method under test
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)

        # Assertions
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next and mode is unknown"""
        # Setup
        number, mode, direction = 1, 'unknown_mode', 'next'

        # Check if ValueError is being raised
        with self.assertRaises(ValueError):
            # Call the method under test
            self.lidar_box.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_unknown_direction(self) -> None:
        """Tests the future_or_past_ego_poses when direction is unknown"""
        # Setup
        number, mode, direction = 1, 'unknown_mode', 'unknown_direction'

        # Check if ValueError is being raised
        with self.assertRaises(ValueError):
            # Call the method under test
            self.lidar_box.future_or_past_ego_poses(number, mode, direction)

    def test_temporal_neighbours_prev_exists(self) -> None:
        """Tests the _temporal_neighbours method when prev exists"""
        # Call the method under test
        result = self.lidar_box._temporal_neighbors()

        # Assertions
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], self.lidar_box.prev)

    def test_temporal_neighbours_prev_is_empty(self) -> None:
        """Tests the _temporal_neighbours method when prev does not exist"""
        # Setup
        # Create a seperate instance of lidar_box for testing
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.prev = None

        # Call the method under test
        result = lidar_box._temporal_neighbors()

        # Assertions
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], lidar_box)

    def test_temporal_neighbours_next_exists(self) -> None:
        """Tests the _temporal_neighbours method when next exists"""
        # Call the method under test
        result = self.lidar_box._temporal_neighbors()

        # Assertions
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1], self.lidar_box.next)

    def test_temporal_neighbours_next_is_empty(self) -> None:
        """Tests the _temporal_neighbours method when next does not exist"""
        # Setup
        # Create a seperate instanc eof lidar_box for testing
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next = None

        # Call the method under test
        result = lidar_box._temporal_neighbors()

        # Assertions
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1], lidar_box)

    def test_velocity_no_next_and_prev(self) -> None:
        """Tests the velocity property when next and prev does not exist"""
        # Setup
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next = None
        lidar_box.prev = None

        # Call the method under test
        result = lidar_box.velocity

        # Assertions
        self.assertTrue(np.isnan(result).any())

    def test_velocity_time_diff_exceed_limit(self) -> None:
        """Tests the velocity property when the difference between timestamps exceed limit"""
        # Setup
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next.lidar_pc.timestamp = lidar_box.prev.lidar_pc.timestamp + 1000000000

        # Call the method under test
        result = lidar_box.velocity

        # Assertions
        self.assertTrue(np.isnan(result).any())

    def test_velocity_default(self) -> None:
        """Tests the default velocity property, should not return any NaN values"""
        # Call the method under test
        result = self.lidar_box.velocity

        # Assertions
        self.assertFalse(np.isnan(result).any())

    def test_angular_velocity_no_next_and_prev(self) -> None:
        """Tests the angular_velocity property when next and prev does not exist"""
        # Setup
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next = None
        lidar_box.prev = None

        # Call the method under test
        result = lidar_box.angular_velocity

        # Assertions
        self.assertTrue(np.isnan(result))

    def test_angular_velocity_time_diff_exceed_limit(self) -> None:
        """Tests the angular_velocity property when the difference between timestamps exceed limit"""
        # Setup
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next.lidar_pc.timestamp = lidar_box.prev.lidar_pc.timestamp + 1000000000

        # Call the method under test
        result = lidar_box.angular_velocity

        # Assertions
        self.assertTrue(np.isnan(result))

    def test_angular_velocity_default(self) -> None:
        """Tests the default angular_velocity property, should not return any NaN values"""
        # Call the method under test
        result = self.lidar_box.angular_velocity

        # Assertions
        self.assertFalse(np.isnan(result))

    def test_box(self) -> None:
        """Tests the box method"""
        # Call the method under test
        result = self.lidar_box.box()

        # Assertions
        self.assertIsInstance(result, Box3D)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.PredictedTrajectory', autospec=True)
    def test_tracked_object_is_agent(self, predicted_trajectory_mock: Mock) -> None:
        """Tests the tracked_object method"""
        # Setup
        future_waypoints = Mock()
        predicted_trajectory_mock.return_value.probability = 1.0

        # Call the method under test
        result = self.lidar_box_vehicle.tracked_object(future_waypoints)

        # Assertions
        predicted_trajectory_mock.assert_called_once_with(1.0, future_waypoints)
        self.assertIsInstance(result, Agent)

    def test_tracked_object_is_static_object(self) -> None:
        """Tests the tracked_object method"""
        # Setup
        future_waypoints = Mock()

        # Call the method under test
        result = self.lidar_box.tracked_object(future_waypoints)

        # Assertions
        self.assertIsInstance(result, StaticObject)

    def test_velocity(self) -> None:
        """Test if velocity is calculated correctly."""
        self.assertTrue(self.lidar_box.prev is not None)
        self.assertTrue(self.lidar_box.next is not None)
        prev_lidar_box: LidarBox = self.lidar_box.prev
        next_lidar_box: LidarBox = self.lidar_box.next

        # Velocity to be tested
        time_diff = 1e-6 * (next_lidar_box.timestamp - prev_lidar_box.timestamp)
        pos_diff = self.lidar_box.velocity * time_diff

        # Next sample position
        pos_next = next_lidar_box.translation_np

        # Predicted next sample position.
        pos_next_pred = prev_lidar_box.translation_np + pos_diff

        # We don't consider velocity in z direction for now
        np.testing.assert_array_almost_equal(pos_next[:2], pos_next_pred[:2], decimal=4)

    def test_angular_velocity(self) -> None:
        """Test if angular velocity is calculated correctly."""
        self.assertTrue(self.lidar_box.prev is not None)
        self.assertTrue(self.lidar_box.next is not None)
        prev_lidar_box: LidarBox = self.lidar_box.prev
        next_lidar_box: LidarBox = self.lidar_box.next

        # Angular velocity to be tested
        time_diff = 1e-6 * (next_lidar_box.timestamp - prev_lidar_box.timestamp)
        yaw_diff = self.lidar_box.angular_velocity * time_diff

        # Previous sample yaw angle.
        yaw_prev = quaternion_yaw(prev_lidar_box.quaternion)
        q_yaw_prev = Quaternion(np.array([np.cos(yaw_prev / 2), 0, 0, np.sin(yaw_prev / 2)]))

        # Predicted next sample yaw angle.
        q_yaw_next_pred = Quaternion(np.array([np.cos(yaw_diff / 2), 0, 0, np.sin(yaw_diff / 2)])) * q_yaw_prev
        yaw_next_pred = quaternion_yaw(q_yaw_next_pred)

        # Next sample yaw angle.
        yaw_next = quaternion_yaw(next_lidar_box.quaternion)

        self.assertAlmostEqual(yaw_next, yaw_next_pred, delta=1e-04)

    def test_next(self) -> None:
        """Test next."""
        self.assertGreater(
            self.lidar_box.next.timestamp,
            self.lidar_box.timestamp,
            "Timestamp of succeeding box must be greater then current box.",
        )

    def test_prev(self) -> None:
        """Test prev."""
        self.assertLess(
            self.lidar_box.prev.timestamp,
            self.lidar_box.timestamp,
            "Timestamp of preceding box must be lower then current box.",
        )

    def test_past_ego_poses(self) -> None:
        """Test if past ego poses are returned correctly."""
        n_ego_poses = 4
        past_ego_poses = self.lidar_box.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = self.lidar_box.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(
                ego_pose.timestamp,
                past_ego_poses[i].timestamp,
                "Timestamp of current EgoPose must be greater than past EgoPoses",
            )

    def test_future_ego_poses(self) -> None:
        """Test if future ego poses are returned correctly."""
        n_ego_poses = 4
        future_ego_poses = self.lidar_box.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.lidar_box.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(
                ego_pose.timestamp,
                future_ego_poses[i].timestamp,
                "Timestamp of current EgoPose must be less than future EgoPoses ",
            )

    def test_get_box_items_to_iterate(self) -> None:
        """Tests the get_box_items_to_iterate method"""
        # Call the method under test
        result = self.lidar_box.get_box_items_to_iterate()

        # Assertions
        # Check if own box timestamp is in the Dict of LidarBoxes
        self.assertTrue(self.lidar_box.timestamp in result)
        self.assertEqual(self.lidar_box.prev, result[self.lidar_box.timestamp][0])
        self.assertEqual(self.lidar_box.next, result[self.lidar_box.timestamp][1])

    @patch('nuplan.database.nuplan_db_orm.lidar_box.IterableLidarBox', autospec=True)
    def test_iter(self, iterable_lidar_box_mock: Mock) -> None:
        """Tests the iterator for LidarBox"""
        # Call the method under test
        result = iter(self.lidar_box)

        # Assertions
        iterable_lidar_box_mock.assert_called_once_with(self.lidar_box)
        self.assertEqual(result, iterable_lidar_box_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.IterableLidarBox', autospec=True)
    def test_reverse_iter(self, iterable_lidar_box_mock: Mock) -> None:
        """Tests the reverse iterator for LidarBox"""
        # Call the method under test
        result = reversed(self.lidar_box)

        # Assertions
        iterable_lidar_box_mock.assert_called_once_with(self.lidar_box, reverse=True)
        self.assertEqual(result, iterable_lidar_box_mock.return_value)


if __name__ == "__main__":
    unittest.main()
