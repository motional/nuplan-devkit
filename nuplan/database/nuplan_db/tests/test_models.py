from pyquaternion import Quaternion
from nuplan.database.utils.geometry import quaternion_yaw
from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.database.nuplan_db.models import LidarBox, generate_multi_scale_connections
import numpy.typing as npt
import numpy as np
import os
import random
import unittest
from typing import Set, Tuple


class TestEgoPose(unittest.TestCase):
    """ Test preparation of point cloud method (standalone method). """

    def setUp(self) -> None:
        self.db = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'),  # type: ignore
                           map_version='nuplan-maps-v0.1')
        self.xrange = (-60, 60)
        self.yrange = (-60, 60)
        self.rotate_face_up = False
        self.map_layer_description = 'intensity'
        self.map_layer_precision = 0.1
        self.map_scale = 1 / self.map_layer_precision
        self.num_samples = 10
        self.selected_indices = random.sample(list(range(len(self.db.ego_pose))), self.num_samples)

    def test_get_map_crop_dimensions(self) -> None:
        """
        Test that map crop method produces map of the correct dimensions.
        Test time: 10.569s
        """
        expected_dimensions = (
            (self.xrange[1] - self.xrange[0]) * self.map_scale,
            (self.yrange[1] - self.yrange[0]) * self.map_scale
        )
        ego_pose_list = self.db.ego_pose
        for i in self.selected_indices:
            current_ego_pose = ego_pose_list[i]
            if current_ego_pose.lidar_pc is None:
                continue
            map_crop = current_ego_pose.get_map_crop(
                xrange=self.xrange,
                yrange=self.yrange,
                map_layer_name=self.map_layer_description,
                rotate_face_up=self.rotate_face_up
            )
            self.assertTrue(map_crop[0] is not None)
            self.assertEqual(expected_dimensions, map_crop[0].shape,  # type: ignore
                             f"Dimensions failed at ego pose index {i}")


class TestImage(unittest.TestCase):
    """ Test Image. """

    def setUp(self) -> None:
        self.db = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'))  # type: ignore
        self.image = self.db.image['b0835c1230135cceac294d2aecc04d00']

    def test_past_ego_poses(self) -> None:
        """ Test if past ego poses are returned correctly. """
        n_ego_poses = 4
        past_ego_poses = self.image.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = self.image.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(ego_pose.timestamp, past_ego_poses[i].timestamp, "Timestamps of current EgoPose "
                                                                                "must be greater than past EgoPoses ")

    def test_future_ego_poses(self) -> None:
        """ Test if future ego poses are returned correctly. """
        n_ego_poses = 4
        future_ego_poses = self.image.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.image.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(ego_pose.timestamp, future_ego_poses[i].timestamp, "Timestamps of current EgoPose "
                                                                               "must be less that future EgoPoses.")


class TestLidarPc(unittest.TestCase):
    """ Test LidarPc. """

    def setUp(self) -> None:
        self.db = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'))  # type: ignore
        self.lidar_pc = self.db.lidar_pc['dcd84209329f5bb0b6f6feb9a3ae155a']

    def test_past_ego_poses(self) -> None:
        """ Test if past ego poses are returned correctly. """
        n_ego_poses = 4
        past_ego_poses = self.lidar_pc.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = self.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(ego_pose.timestamp, past_ego_poses[i].timestamp, "Timestamps of current EgoPose "
                                                                                "must be greater than past EgoPoses.")

    def test_future_ego_poses(self) -> None:
        """ Test if future ego poses are returned correctly. """
        n_ego_poses = 4
        future_ego_poses = self.lidar_pc.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(ego_pose.timestamp, future_ego_poses[i].timestamp, "Timestamps of current EgoPose "
                                                                               "must be less that future EgoPoses.")


class TestLidarBox(unittest.TestCase):
    """ Test LidarBox Annotation. """

    def setUp(self) -> None:
        self.db = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'))  # type: ignore
        self.lidar_box = self.db.lidar_box[1234567]

    def test_velocity(self) -> None:
        """ Test if velocity is calculated correctly. """

        self.assertTrue(self.lidar_box.prev is not None)
        self.assertTrue(self.lidar_box.next is not None)
        prev_lidar_box: LidarBox = self.lidar_box.prev  # type: ignore
        next_lidar_box: LidarBox = self.lidar_box.next  # type: ignore

        # Velocity to be tested
        time_diff = 1e-6 * (next_lidar_box.timestamp - prev_lidar_box.timestamp)
        pos_diff = self.lidar_box.velocity * time_diff  # type: ignore

        # Next sample position
        pos_next = next_lidar_box.translation_np

        # Predicted next sample position.
        pos_next_pred = prev_lidar_box.translation_np + pos_diff  # type: ignore

        # We don't consider velocity in z direction for now
        np.testing.assert_array_almost_equal(pos_next[:2], pos_next_pred[:2], decimal=4)  # type: ignore

    def test_angular_velocity(self) -> None:
        """ Test if angular velocity is calculated correctly. """
        self.assertTrue(self.lidar_box.prev is not None)
        self.assertTrue(self.lidar_box.next is not None)
        prev_lidar_box: LidarBox = self.lidar_box.prev  # type: ignore
        next_lidar_box: LidarBox = self.lidar_box.next  # type: ignore

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
        """ Test next. """
        self.assertGreater(self.lidar_box.next.timestamp, self.lidar_box.timestamp,  # type: ignore
                           "Timestamp of succeeding box must be greater then current box.")

    def test_prev(self) -> None:
        """ Test prev. """
        self.assertLess(self.lidar_box.prev.timestamp, self.lidar_box.timestamp,  # type: ignore
                        "Timestamp of preceding box must be lower then current box.")

    def test_past_ego_poses(self) -> None:
        """ Test if past ego poses are returned correctly. """
        n_ego_poses = 4
        past_ego_poses = self.lidar_box.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = self.lidar_box.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(ego_pose.timestamp, past_ego_poses[i].timestamp, "Timestamp of current EgoPose must be "
                                                                                "greater than past EgoPoses")

    def test_future_ego_poses(self) -> None:
        """ Test if future ego poses are returned correctly. """
        n_ego_poses = 4
        future_ego_poses = self.lidar_box.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.lidar_box.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(ego_pose.timestamp, future_ego_poses[i].timestamp, "Timestamp of current EgoPose must be"
                                                                               "less than future EgoPoses ")


class TestTrack(unittest.TestCase):
    """ Test Track. """

    def setUp(self) -> None:
        self.db = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'))  # type: ignore
        self.track = self.db.track[123]

    def test_nbr_lidar_boxes(self) -> None:
        """ Test number of annotations in a track. """
        self.assertGreater(self.track.nbr_lidar_boxes, 0)

    def test_duration(self) -> None:
        """ Test duration of a track. """
        self.assertGreater(self.track.duration, 0)

    def test_first_last_lidar_box(self) -> None:
        """ Test first and lsat boxes in a track. """
        first_lidar_box = self.track.first_lidar_box
        last_lidar_box = self.track.last_lidar_box

        self.assertGreaterEqual(last_lidar_box.timestamp, first_lidar_box.timestamp)

    def test_distance_to_ego(self) -> None:
        """ Test distance to ego vehicle in a track. """
        self.assertGreaterEqual(self.track.max_distance_to_ego, self.track.min_distance_to_ego)


class TestGenerateMultiScaleConnections(unittest.TestCase):

    def test_generate_multi_scale_connections(self) -> None:
        """ Test generate_multi_scale_connections() """
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
