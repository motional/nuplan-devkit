import random
import unittest
from typing import Set, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.database.nuplan_db.lidar_box import LidarBox
from nuplan.database.nuplan_db.models import generate_multi_scale_connections
from nuplan.database.tests.nuplan_db_test_utils import (
    get_test_nuplan_db,
    get_test_nuplan_image,
    get_test_nuplan_lidar_box,
    get_test_nuplan_lidarpc,
)
from nuplan.database.utils.geometry import quaternion_yaw
from pyquaternion import Quaternion


@unittest.skip('No nuPlan sensor blobs currently available in Jenkins')
class TestImage(unittest.TestCase):
    """Test Image."""

    def setUp(self) -> None:
        self.image = get_test_nuplan_image()

    def test_past_ego_poses(self) -> None:
        """Test if past ego poses are returned correctly."""
        n_ego_poses = 4
        past_ego_poses = self.image.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = self.image.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(
                ego_pose.timestamp,
                past_ego_poses[i].timestamp,
                "Timestamps of current EgoPose must be greater than past EgoPoses ",
            )

    def test_future_ego_poses(self) -> None:
        """Test if future ego poses are returned correctly."""
        n_ego_poses = 4
        future_ego_poses = self.image.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.image.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(
                ego_pose.timestamp,
                future_ego_poses[i].timestamp,
                "Timestamps of current EgoPose must be less that future EgoPoses.",
            )


class TestGenerateMultiScaleConnections(unittest.TestCase):
    def test_generate_multi_scale_connections(self) -> None:
        """Test generate_multi_scale_connections()"""
        connections = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 3], [2, 4]])
        scales = [1, 2, 4]
        expected_multi_scale_connections = {
            1: connections,
            2: np.array([[0, 2], [1, 3], [2, 4], [3, 5], [0, 4], [1, 4], [2, 5]]),
            4: np.array([[0, 4], [0, 5], [1, 5]]),
        }
        multi_scale_connections = generate_multi_scale_connections(connections, scales)

        def _convert_to_connection_set(connection_array: npt.NDArray[np.float64]) -> Set[Tuple[float, float]]:
            """
            Convert connections from array to set.

            :param connection_array: <np.float: N, 2>. Connection in array format.
            :return: Connection in set format.
            """
            return {(connection[0], connection[1]) for connection in connection_array}

        self.assertEqual(multi_scale_connections.keys(), expected_multi_scale_connections.keys())
        for key in multi_scale_connections:
            connection_set = _convert_to_connection_set(multi_scale_connections[key])
            expected_connection_set = _convert_to_connection_set(expected_multi_scale_connections[key])
            self.assertEqual(connection_set, expected_connection_set)


if __name__ == '__main__':
    unittest.main()
