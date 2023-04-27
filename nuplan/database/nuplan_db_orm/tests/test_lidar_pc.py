import unittest
from copy import deepcopy
from unittest.mock import Mock, PropertyMock, patch

from matplotlib.axes import Axes

from nuplan.database.nuplan_db_orm.log import Log
from nuplan.database.tests.test_utils_nuplan_db import (
    get_test_nuplan_db,
    get_test_nuplan_lidarpc,
    get_test_nuplan_lidarpc_with_blob,
)


class TestLidarPc(unittest.TestCase):
    """Tests the LidarBox class"""

    def setUp(self) -> None:
        """Sets up for the tests cases"""
        self.lidar_pc = get_test_nuplan_lidarpc()
        self.lidar_pc_with_blob = get_test_nuplan_lidarpc_with_blob()

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the _session property"""
        # Setup
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock

        # Call the method under test
        result = self.lidar_pc._session

        # Assertions
        inspect_mock.assert_called_once_with(self.lidar_pc)
        self.assertEqual(result, session_mock)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the __repr__ method"""
        # Call the method under test
        result = self.lidar_pc.__repr__()

        # Assertions
        simple_repr_mock.assert_called_once_with(self.lidar_pc)
        self.assertEqual(result, simple_repr_mock.return_value)

    def test_log(self) -> None:
        """Tests the log property"""
        # Call the method under test
        result = self.lidar_pc.log

        # Assertions
        self.assertIsInstance(result, Log)

    def test_future_ego_pose_has_next(self) -> None:
        """Tests the future_ego_pose method when there is a future ego pose"""
        # Call the method under test
        result = self.lidar_pc.future_ego_pose()

        # Assertions
        self.assertEqual(result, self.lidar_pc.next.ego_pose)

    def test_future_ego_pose_no_next(self) -> None:
        """Tests the future_ego_pose method when there is no future ego pose"""
        # Setup
        lidar_pc = deepcopy(self.lidar_pc)
        lidar_pc.next = None

        # Call the method under test
        result = lidar_pc.future_ego_pose()

        # Assertions
        self.assertEqual(result, None)

    def test_past_ego_pose_has_prev(self) -> None:
        """Tests the past_ego_pose method when there is a past ego pose"""
        # Call the method under test
        result = self.lidar_pc.past_ego_pose()

        # Assertions
        self.assertEqual(result, self.lidar_pc.prev.ego_pose)

    def test_past_ego_pose_no_prev(self) -> None:
        """Tests the past_ego_pose method when there is no past ego pose"""
        # Setup
        lidar_pc = deepcopy(self.lidar_pc)
        lidar_pc.prev = None

        # Call the method under test
        result = lidar_pc.past_ego_pose()

        # Assertions
        self.assertEqual(result, None)

    def test_future_or_past_ego_poses_prev_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_poses"""
        # Setup
        number, mode, direction = 1, 'n_poses', 'prev'

        # Call the method under test
        result = self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

        # Assertions
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_seconds"""
        # Setup
        number, mode, direction = 1, 'n_seconds', 'prev'

        # Call the method under test
        result = self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

        # Assertions
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev and mode is unknown"""
        # Setup
        number, mode, direction = 1, 'unknown_mode', 'prev'

        # Call the method under test
        with self.assertRaises(ValueError):
            # Should result in a ValueError being raised
            self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_next_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_poses"""
        # Setup
        number, mode, direction = 1, 'n_poses', 'next'

        # Call the method under test
        result = self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

        # Assertions
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_seconds"""
        # Setup
        number, mode, direction = 1, 'n_seconds', 'next'

        # Call the method under test
        result = self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

        # Assertions
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next and mode is unknown"""
        # Setup
        number, mode, direction = 1, 'unknown_mode', 'next'

        # Check if ValueError is being raised
        with self.assertRaises(ValueError):
            # Call the method under test
            self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_unknown_direction(self) -> None:
        """Tests the future_or_past_ego_poses when direction is unknown"""
        # Setup
        number, mode, direction = 1, 'unknown_mode', 'unknown_direction'

        # Check if ValueError is being raised
        with self.assertRaises(ValueError):
            # Call the method under test
            self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.LidarPointCloud.from_buffer', autospec=True)
    def test_load_channel_is_merged_point_cloud(self, from_buffer_mock: Mock) -> None:
        """Tests the load method when lidar channel is MergedPointCloud"""
        # Setup
        db = get_test_nuplan_db()

        # Call the method under test
        result = self.lidar_pc.load(db)

        # Assertions
        self.assertEqual(result, from_buffer_mock.return_value)

    def test_load_channel_is_not_implemented(self) -> None:
        """Tests the load method when lidar channel is not implemented"""
        # Setup
        db = get_test_nuplan_db()
        lidar_pc = deepcopy(self.lidar_pc)
        lidar_pc.lidar.channel = 'UnknownPointCloud'

        # Should raise a NotImplementedError
        with self.assertRaises(NotImplementedError):
            # Call the method under test
            lidar_pc.load(db)

    def test_load_bytes(self) -> None:
        """Tests the load bytes method"""
        # Setup
        db = get_test_nuplan_db()

        # Call the method under test
        result = self.lidar_pc_with_blob.load_bytes(db)

        # Assertions
        self.assertIsNotNone(result)

    def test_path(self) -> None:
        """Tests the path property"""
        # Setup
        db = get_test_nuplan_db()

        # Call the method under test
        result = self.lidar_pc_with_blob.path(db)

        # Assertions
        self.assertIsInstance(result, str)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.get_boxes', autospec=True)
    def test_boxes(self, get_boxes_mock: Mock) -> None:
        """Tests the boxes method"""
        # Call the method under test
        result = self.lidar_pc.boxes()

        # Assertions
        self.assertEqual(result, get_boxes_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.pack_future_boxes', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.lidar_pc.get_future_box_sequence', autospec=True)
    def test_boxes_with_future_waypoints(
        self,
        get_future_box_sequence_mock: Mock,
        pack_future_boxes_mock: Mock,
    ) -> None:
        """Tests the boxes_with_future_waypoints method"""
        # Setup
        future_horizon_len_s, future_interval_s = 1.0, 1.0

        # Call the method under test
        result = self.lidar_pc.boxes_with_future_waypoints(future_horizon_len_s, future_interval_s)

        # Assertions
        get_future_box_sequence_mock.assert_called_once()
        pack_future_boxes_mock.assert_called_once_with(
            get_future_box_sequence_mock.return_value, future_interval_s, future_horizon_len_s
        )
        self.assertEqual(result, pack_future_boxes_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.render_on_map', autospec=True)
    def test_render(self, render_on_map_mock: Mock) -> None:
        """Tests the render method"""
        # Setup
        db = get_test_nuplan_db()

        # Call the method under test
        result = self.lidar_pc_with_blob.render(db)

        # Assertions
        render_on_map_mock.assert_called_once()
        self.assertIsInstance(result, Axes)

    def test_past_ego_poses(self) -> None:
        """Test if past ego poses are returned correctly."""
        n_ego_poses = 4
        lidar_pc = self.lidar_pc.next.next.next
        past_ego_poses = lidar_pc.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(
                ego_pose.timestamp,
                past_ego_poses[i].timestamp,
                "Timestamps of current EgoPose must be greater than past EgoPoses.",
            )

    def test_future_ego_poses(self) -> None:
        """Test if future ego poses are returned correctly."""
        n_ego_poses = 4
        future_ego_poses = self.lidar_pc.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(
                ego_pose.timestamp,
                future_ego_poses[i].timestamp,
                "Timestamps of current EgoPose must be less that future EgoPoses.",
            )


if __name__ == '__main__':
    unittest.main()
