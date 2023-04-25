import unittest
from typing import List, Optional, Set, Tuple
from unittest.mock import MagicMock, Mock

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.utils import (
    _get_past_future_sweep,
    generate_multi_scale_connections,
    get_boxes,
    get_future_box_sequence,
    get_future_ego_trajectory,
    load_boxes_from_lidarpc,
    load_pointcloud_from_pc,
    pack_future_boxes,
    prepare_pointcloud_points,
    render_on_map,
)
from nuplan.database.tests.test_utils_nuplan_db import (
    get_test_nuplan_db,
    get_test_nuplan_lidarpc,
    get_test_nuplan_lidarpc_with_blob,
)
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud


class TestGenerateMultiScaleConnections(unittest.TestCase):
    """
    Test generation of multi-scale connections
    """

    def test_generate_multi_scale_connections(self) -> None:
        """Test generate_multi_scale_connections()"""
        connections = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 3], [2, 4]], dtype=np.float64
        )  # type: npt.NDArray[np.float64]
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


class TestGetBoxes(unittest.TestCase):
    """Test get box."""

    # TODO: Setup map

    # Helper data ##############################################################

    def _box_A(self) -> Box3D:
        """
        Helper method to get one box.
        :return: One box.
        """
        return Box3D(
            center=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            orientation=Quaternion(axis=[1, 0, 0], angle=0),
            velocity=(0.0, 0.0, 0.0),
            angular_velocity=0.0,
        )

    def _box_B(self) -> Box3D:
        """
        Helper method to get one box.
        :return: One box.
        """
        return Box3D(
            center=(1.0, 2.0, 3.0),
            size=(1.0, 1.0, 1.0),
            orientation=Quaternion(axis=[1, 0, 0], angle=2),
            velocity=(5.0, 6.0, 7.0),
            angular_velocity=8.0,
        )

    def _box_quarterway_between_A_and_B(self) -> Box3D:
        """
        Helper method to get one box.
        :return: One box.
        """
        return Box3D(
            center=(0.25, 0.5, 0.75),
            size=(1.0, 1.0, 1.0),
            orientation=Quaternion(axis=[1, 0, 0], angle=0.5),
            velocity=(1.25, 1.5, 1.75),
            angular_velocity=2.0,
        )

    def _box_halfway_between_A_and_B(self) -> Box3D:
        """
        Helper method to get one box.
        :return: One box.
        """
        return Box3D(
            center=(0.5, 1.0, 1.5),
            size=(1.0, 1.0, 1.0),
            orientation=Quaternion(axis=[1, 0, 0], angle=1),
            velocity=(2.5, 3, 3.5),
            angular_velocity=4.0,
        )

    def _annotation_A(self, track_token: str) -> Mock:
        """
        Helper method to get one annotation.
        :param track_token: Track token to use.
        :return: Mocked annotation.
        """
        ann = Mock()
        ann.x = 0.0
        ann.y = 0.0
        ann.z = 0.0
        ann.translation_np = np.array([ann.x, ann.y, ann.z])
        ann.width = 1.0
        ann.length = 1.0
        ann.height = 1.0
        ann.size = (ann.width, ann.length, ann.height)
        ann.roll = 0.0
        ann.pitch = 0.0
        ann.yaw = 0.0
        ann.quaternion = Quaternion(axis=[1, 0, 0], angle=0)
        ann.vx = 0.0
        ann.vy = 0.0
        ann.vz = 0.0
        ann.velocity = np.array([ann.vx, ann.vy, ann.vz])
        ann.angular_velocity = 0.0
        ann.box.return_value = self._box_A()
        ann.track_token = track_token
        return ann

    def _annotation_B(self, track_token: str) -> Mock:
        """
        Helper method to get one annotation.
        :param track_token: Track token to use.
        :return: Mocked annotation.
        """
        ann = Mock()
        ann.x = 1.0
        ann.y = 2.0
        ann.z = 3.0
        ann.translation_np = np.array([ann.x, ann.y, ann.z])
        ann.width = 1.0
        ann.length = 1.0
        ann.height = 1.0
        ann.size = (ann.width, ann.length, ann.height)
        ann.roll = 0.0
        ann.pitch = 0.0
        ann.yaw = 0.0
        ann.quaternion = Quaternion(axis=[1, 0, 0], angle=2)
        ann.vx = 5.0
        ann.vy = 6.0
        ann.vz = 7.0
        ann.velocity = np.array([ann.vx, ann.vy, ann.vz])
        ann.angular_velocity = 8.0
        ann.box.return_value = self._box_B()
        ann.track_token = track_token
        return ann

    def _trans_matrix_ego(self) -> npt.NDArray[np.float64]:
        """
        Helper method to get a transformation.
        :return: <np.float: 4, 4> Transformation matrix.
        """
        # These trans matrices aren't realistic, they just need to be orthogonal
        # affine transforms.
        return np.array([[0, 1, 0, 1], [-1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])

    def _trans_matrix_sensor(self) -> npt.NDArray[np.float64]:
        """
        Helper method to get a transformation.
        :return: <np.float: 4, 4> Transformation matrix.
        """
        return np.array([[0, 0, 1, 4], [0, -1, 0, 5], [1, 0, 0, 6], [0, 0, 0, 1]])

    # Tests ####################################################################

    def test_frame_vehicle(self) -> None:
        """
        Test putting resulting boxes in vehicle coordinates.
        """
        lidarpc = Mock()
        lidarpc.lidar_boxes = [self._annotation_B(track_token="456")]
        lidarpc.prev = object()  # Just needs to be non-None

        box_b_vehicle_frame = self._box_B()
        box_b_vehicle_frame.transform(self._trans_matrix_ego())

        self.assertEqual(
            get_boxes(lidarpc, frame=Frame.VEHICLE, trans_matrix_ego=self._trans_matrix_ego()), [box_b_vehicle_frame]
        )

    def test_frame_sensor(self) -> None:
        """
        Test putting resulting boxes in sensor coordinates.
        """
        lidarpc = Mock()
        lidarpc.lidar_boxes = [self._annotation_B(track_token="456")]
        lidarpc.prev = object()  # Just needs to be non-None

        box_b_sensor_frame = self._box_B()
        box_b_sensor_frame.transform(self._trans_matrix_ego())
        box_b_sensor_frame.transform(self._trans_matrix_sensor())

        self.assertEqual(
            get_boxes(
                lidarpc,
                frame=Frame.SENSOR,
                trans_matrix_ego=self._trans_matrix_ego(),
                trans_matrix_sensor=self._trans_matrix_sensor(),
            ),
            [box_b_sensor_frame],
        )


class TestPointCloudPreparation(unittest.TestCase):
    """Test preparation of point cloud method (standalone method)."""

    def setUp(self) -> None:
        """Setup funciton for class."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()

    def test_prepare_pointcloud_points(self) -> None:
        """
        Tests if the lidar point clouds are properly filtered when loaded and decorations are correctly applied.
        """
        pc_none = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=False,
            use_ring=False,
            use_lidar_index=False,
            lidar_indices=None,
            sample_apillar_lidar_rings=False,
        )

        pc_intensity = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=True,
            use_ring=False,
            use_lidar_index=False,
            lidar_indices=None,
            sample_apillar_lidar_rings=False,
        )

        pc_ring = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=False,
            use_ring=True,
            use_lidar_index=False,
            lidar_indices=None,
            sample_apillar_lidar_rings=False,
        )

        pc_lidar = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=False,
            use_ring=False,
            use_lidar_index=True,
            lidar_indices=None,
            sample_apillar_lidar_rings=False,
        )

        pc_all = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=True,
            use_ring=True,
            use_lidar_index=True,
            lidar_indices=None,
            sample_apillar_lidar_rings=False,
        )

        pc_single_lidar = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=True,
            use_ring=True,
            use_lidar_index=True,
            lidar_indices=(0,),
            sample_apillar_lidar_rings=True,
        )

        pc_all_0 = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=True,
            use_ring=True,
            use_lidar_index=True,
            lidar_indices=(0, 1, 2, 3, 4),
            sample_apillar_lidar_rings=True,
        )

        pc_all_1 = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=True,
            use_ring=True,
            use_lidar_index=True,
            lidar_indices=None,
            sample_apillar_lidar_rings=True,
        )

        pc_all_no_sample_apillar = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=True,
            use_ring=True,
            use_lidar_index=True,
            lidar_indices=None,
            sample_apillar_lidar_rings=False,
        )

        pc_sample_apillar_0 = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=True,
            use_ring=True,
            use_lidar_index=True,
            lidar_indices=(0, 3, 4),
            sample_apillar_lidar_rings=True,
        )

        pc_sample_apillar_1 = prepare_pointcloud_points(
            self.lidar_pc.load(self.db),
            use_intensity=True,
            use_ring=True,
            use_lidar_index=True,
            lidar_indices=(0, 3, 4),
            sample_apillar_lidar_rings=False,
        )

        pt_cloud = self.lidar_pc.load(self.db)

        # Check that the decorations (i.e. intensity, ring, lidar indices) are correctly applied.
        self.assertEqual(pc_none.points.shape[0], 3)
        self.assertEqual(pc_intensity.points.shape[0], 4)
        self.assertEqual(pc_ring.points.shape[0], 4)
        self.assertEqual(pc_lidar.points.shape[0], 4)
        self.assertEqual(pc_all.points.shape[0], 6)

        # Check that the original merged point clouds with multiple lidars has more points than single lidar.
        self.assertTrue(pt_cloud.nbr_points() > pc_single_lidar.nbr_points())

        # Check lidar_indicies (None and (0,1,2,3,4)) are equivalent.
        self.assertTrue((pc_all_0.points == pc_all_1.points).all())

        # Check that point clouds are only from the given lidar indices.
        self.assertTrue((pc_single_lidar.points[5] == 0).all())
        self.assertTrue(np.isin(pc_all_0.points[5], [0, 1, 2, 3, 4]).all())

        # Check sampling of A-pillar indices has no effect when the two A-pillar lidar indices are not provided.
        self.assertTrue(np.array_equal(pc_sample_apillar_0.points, pc_sample_apillar_1.points))

        # Compare sizes for non-sampling/sampling of A-pillar lidars (lidar indices 1 and 2).
        self.assertTrue(
            pc_all_no_sample_apillar.points[
                :, np.logical_or(pc_all_no_sample_apillar.points[5] == 1, pc_all_no_sample_apillar.points[5] == 2)
            ].shape[1]
            > pc_all_0.points[:, np.logical_or(pc_all_0.points[5] == 1, pc_all_0.points[5] == 2)].shape[1]
        )


class TestNuPlanDBLidarMethods(unittest.TestCase):
    """Tests for NuPlanDBLidarMethods (Helper methods for interacting with NuPlanDB's lidar samples)."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()

    def test_get_past_future_sweep(self) -> None:
        """
        Go N sweeps back and N sweeps forth and see if we are back at the original.
        """
        for sweep_idx in range(-10, 10):
            sweep_lidarpc_rec = _get_past_future_sweep(self.lidar_pc, sweep_idx)
            if sweep_lidarpc_rec is None:
                continue  # If we hit the start or end of a scene this test should be skipped.
            return_lidarpc_rec = _get_past_future_sweep(sweep_lidarpc_rec, -sweep_idx)
            self.assertEqual(self.lidar_pc, return_lidarpc_rec)

    def test_load_pointcloud_from_pc(self) -> None:
        """
        Test loading of point cloud from LidarPc based on distance, data shape, map filtering and timestmap.
        """
        min_dist = 0.9
        max_dist = 50.0

        pc = load_pointcloud_from_pc(
            nuplandb=self.db,
            token=self.lidar_pc.token,
            nsweeps=1,
            max_distance=max_dist,
            min_distance=min_dist,
            use_intensity=False,
            use_ring=False,
            use_lidar_index=False,
        )

        pc_intensity = load_pointcloud_from_pc(
            nuplandb=self.db,
            token=self.lidar_pc.token,
            nsweeps=1,
            max_distance=max_dist,
            min_distance=min_dist,
            use_intensity=True,
            use_ring=False,
            use_lidar_index=False,
        )

        pc_ring = load_pointcloud_from_pc(
            nuplandb=self.db,
            token=self.lidar_pc.token,
            nsweeps=1,
            max_distance=max_dist,
            min_distance=min_dist,
            use_intensity=False,
            use_ring=True,
            use_lidar_index=False,
        )

        pc_lidar_index = load_pointcloud_from_pc(
            nuplandb=self.db,
            token=self.lidar_pc.token,
            nsweeps=1,
            max_distance=max_dist,
            min_distance=min_dist,
            use_intensity=False,
            use_ring=False,
            use_lidar_index=True,
        )

        pc_multiple_sweeps = load_pointcloud_from_pc(
            nuplandb=self.db,
            token=self.lidar_pc.token,
            nsweeps=3,
            max_distance=max_dist,
            min_distance=min_dist,
            use_intensity=True,
            use_ring=False,
            use_lidar_index=False,
        )

        pc_multiple_sweeps_new_format = load_pointcloud_from_pc(
            nuplandb=self.db,
            token=self.lidar_pc.token,
            nsweeps=list(range(-3 + 1, 0 + 1)),
            max_distance=max_dist,
            min_distance=min_dist,
            use_intensity=True,
            use_ring=False,
            use_lidar_index=False,
        )

        pc_map_filtered_random = load_pointcloud_from_pc(
            nuplandb=self.db,
            token=self.lidar_pc.token,
            nsweeps=1,
            max_distance=max_dist,
            min_distance=min_dist,
            use_intensity=False,
            use_ring=False,
            use_lidar_index=False,
        )

        pc_past_future = load_pointcloud_from_pc(
            nuplandb=self.db,
            token=self.lidar_pc.token,
            nsweeps=[-2, 0, 1],
            max_distance=max_dist,
            min_distance=min_dist,
            use_intensity=False,
            use_ring=False,
            use_lidar_index=False,
        )

        pc_dist_from_orig = np.linalg.norm(pc.points[:2, :], axis=0)
        pc_multiple_sweeps_dist_from_orig = np.linalg.norm(pc_multiple_sweeps.points[:2, :], axis=0)

        # Check point cloud has time, x, y, z (4 elements).
        self.assertEqual(pc.points.shape[0], 4)
        self.assertEqual(pc_map_filtered_random.points.shape[0], 4)

        # Check point cloud has time, x, y, z, intensity/ring/lidar_index (5 elements).
        self.assertEqual(pc_intensity.points.shape[0], 5)
        self.assertEqual(pc_ring.points.shape[0], 5)
        self.assertEqual(pc_lidar_index.points.shape[0], 5)
        self.assertEqual(pc_multiple_sweeps.points.shape[0], 5)

        # Check point cloud distance is truly bounded between max and min distance.
        self.assertTrue((pc_dist_from_orig >= min_dist).all() and (pc_dist_from_orig <= max_dist).all())

        # For multiple sweeps we only check the maximum distance, since minimum distance is enforced per sweep.
        self.assertTrue((pc_multiple_sweeps_dist_from_orig <= max_dist).all())

        # Check multi sweep point cloud has at least as many points that that of single sweep.
        self.assertTrue(pc_multiple_sweeps.points.shape[1] >= pc.points.shape[1])

        # Check no. of points in point cloud are reduced for filtered point clouds.
        self.assertTrue(pc_map_filtered_random.points.shape[1] <= pc.points.shape[1])

        # Check that past and future sweeps are included and striding works by looking at the timestamps
        timestamps = np.unique(pc_past_future.points[3, :])  # type: ignore
        past_timestamp = (self.lidar_pc.timestamp - self.lidar_pc.prev.prev.timestamp) / 1e6
        future_timestamp = (self.lidar_pc.timestamp - self.lidar_pc.next.timestamp) / 1e6
        self.assertAlmostEqual(past_timestamp, timestamps[2])  # Timestamps are sorted in ascending order
        self.assertAlmostEqual(0, timestamps[1])
        self.assertAlmostEqual(future_timestamp, timestamps[0])
        self.assertTrue(len(timestamps) == 3)

        # Check that the new list format for nsweeps returns the same as the old format
        self.assertTrue(np.all(pc_multiple_sweeps.points == pc_multiple_sweeps_new_format.points))


def mock_prepare_pointcloud_points(
    pc: LidarPointCloud,
    use_intensity: bool = True,
    use_ring: bool = False,
    use_lidar_index: bool = False,
    lidar_indices: Optional[Tuple[int, ...]] = None,
    sample_apillar_lidar_rings: bool = False,
) -> LidarPointCloud:
    """
    Mock Pointcloud points.
    :param pc: Pointcloud input.
    :param use_intensity: Whether to use intensity or not.
    :param use_ring: Whether to use ring index or not.
    :param use_lidar_index: Whether to use lidar index as a decoration.
    :param lidar_indices: Which lidars to keep.
        MergedPointCloud has following options:
            0: top lidar
            1: right A pillar lidar
            2: left A pillar lidar
            3: back lidar
            4: front lidar
            None: Use all lidars
    :param sample_apillar_lidar_rings: Whether you want to sample rings for the A-pillar lidars.
    """
    return pc


class TestLoadPointcloudFromSampledataUsingMocks(unittest.TestCase):
    """Test Loading PointCloud."""

    @unittest.mock.patch("nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points")
    def test_distance_filtering(self, prepare_pointcloud_points_mock: Mock) -> None:
        """
        Make sure close and far points are filtered properly.
        """
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(
            points=np.array([[0.1, -0.1, 10, -10, 1000, 1000], [0.2, -0.2, 20, 20, 2000, -2000]])
        )

        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(nuplandb, token="abc", nsweeps=1, max_distance=1000, min_distance=1)
        expected_points = np.array([[10, -10], [20, 20], [0, 0]], dtype=np.float32)  # type: ignore

        self.assertTrue(np.allclose(loaded_pc.points, expected_points))

    @unittest.mock.patch("nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points")
    def test_3_sweeps(self, prepare_pointcloud_points_mock: Mock) -> None:
        """
        Make sure points and timestamps accumulate properly with multiple sweeps.
        """
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(points=np.array([[100, -100], [200, 200], [300, 300]]))
        mock_lidarpc_rec.prev.load.return_value = LidarPointCloud(points=np.array([[10, -10], [20, 20], [30, 30]]))
        mock_lidarpc_rec.prev.prev.load.return_value = LidarPointCloud(points=np.array([[1, -1], [2, 2], [3, 3]]))

        mock_lidarpc_rec.timestamp = 507
        mock_lidarpc_rec.prev.timestamp = 504
        mock_lidarpc_rec.prev.prev.timestamp = 500

        mock_lidarpc_rec.lidar.trans_matrix = np.eye(4)
        mock_lidarpc_rec.lidar.trans_matrix_inv = np.eye(4)

        mock_lidarpc_rec.ego_pose.trans_matrix_inv = np.eye(4)
        mock_lidarpc_rec.prev.ego_pose.trans_matrix = np.eye(4)
        mock_lidarpc_rec.prev.prev.ego_pose.trans_matrix = np.eye(4)

        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(nuplandb, token="abc", nsweeps=3, max_distance=1000, min_distance=0)
        expected_points = np.array(
            [
                [1, -1, 10, -10, 100, -100],
                [2, 2, 20, 20, 200, 200],
                [3, 3, 30, 30, 300, 300],
                [7e-6, 7e-6, 3e-6, 3e-6, 0, 0],
            ],
            dtype=np.float32,
        )  # type: ignore

        self.assertTrue(np.allclose(loaded_pc.points, expected_points))

    @unittest.mock.patch("nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points")
    def test_3_sweeps_past_future(self, prepare_pointcloud_points_mock: Mock) -> None:
        """
        Make sure points and timestamps accumulate properly with multiple sweeps, using past and future data.
        """
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(points=np.array([[100, -100], [200, 200], [300, 300]]))
        mock_lidarpc_rec.next.next.load.return_value = LidarPointCloud(points=np.array([[10, -10], [20, 20], [30, 30]]))
        mock_lidarpc_rec.prev.prev.load.return_value = LidarPointCloud(points=np.array([[1, -1], [2, 2], [3, 3]]))

        mock_lidarpc_rec.prev.prev.timestamp = 500
        mock_lidarpc_rec.timestamp = 504
        mock_lidarpc_rec.next.next.timestamp = 507

        mock_lidarpc_rec.lidar.trans_matrix = np.eye(4)
        mock_lidarpc_rec.lidar.trans_matrix_inv = np.eye(4)

        mock_lidarpc_rec.ego_pose.trans_matrix_inv = np.eye(4)
        mock_lidarpc_rec.prev.prev.ego_pose.trans_matrix = np.eye(4)
        mock_lidarpc_rec.next.next.ego_pose.trans_matrix = np.eye(4)

        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(
            nuplandb, token="abc", nsweeps=[-2, 0, 2], max_distance=1000, min_distance=0
        )
        expected_points = np.array(
            [
                [1, -1, 100, -100, 10, -10],
                [2, 2, 200, 200, 20, 20],
                [3, 3, 300, 300, 30, 30],
                [4e-6, 4e-6, 0, 0, -3e-6, -3e-6],
            ],
            dtype=np.float32,
        )  # type: ignore
        self.assertTrue(np.allclose(loaded_pc.points, expected_points))

    @unittest.mock.patch("nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points")
    def test_5_sweeps_moving_vehicle(self, prepare_pointcloud_points_mock: Mock) -> None:
        """Test accumulating sweeps with moving vehicle."""
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points

        # The car observes a single point at position (1, 1, 1) in each frame, but the
        # car itself will be in a different position for each frame.
        point_111 = np.ones((3, 1), dtype=np.float32)  # type: ignore
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(points=point_111)
        mock_lidarpc_rec.prev.load.return_value = LidarPointCloud(points=point_111)
        mock_lidarpc_rec.prev.prev.load.return_value = LidarPointCloud(points=point_111)
        mock_lidarpc_rec.prev.prev.prev.load.return_value = LidarPointCloud(points=point_111)
        mock_lidarpc_rec.prev.prev.prev.prev.load.return_value = LidarPointCloud(points=point_111)

        mock_lidarpc_rec.timestamp = 504
        mock_lidarpc_rec.prev.timestamp = 503
        mock_lidarpc_rec.prev.prev.timestamp = 502
        mock_lidarpc_rec.prev.prev.prev.timestamp = 501
        mock_lidarpc_rec.prev.prev.prev.prev.timestamp = 500

        mock_lidarpc_rec.lidar.trans_matrix = np.eye(4)
        mock_lidarpc_rec.lidar.trans_matrix_inv = np.eye(4)

        def addition_transform(x: float, y: float, z: float) -> npt.NDArray[np.float64]:
            """
            Create a 4 by 4 transformation matrix given translation.
            :return: <np.float: 4, 4>. The transformation matrix.
            """
            return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float32)

        # final car position is at the origin
        mock_lidarpc_rec.ego_pose.trans_matrix_inv = np.eye(4)
        # car's position one frame ago was (1, 2, 3), etc.
        mock_lidarpc_rec.prev.ego_pose.trans_matrix = addition_transform(1, 2, 3)
        mock_lidarpc_rec.prev.prev.ego_pose.trans_matrix = addition_transform(2, 3, 4)
        mock_lidarpc_rec.prev.prev.prev.ego_pose.trans_matrix = addition_transform(3, 4, 5)
        mock_lidarpc_rec.prev.prev.prev.prev.ego_pose.trans_matrix = addition_transform(4, 5, 6)

        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(nuplandb, token="abc", nsweeps=5, max_distance=1000, min_distance=0)
        expected_points = np.array(
            [[5, 4, 3, 2, 1], [6, 5, 4, 3, 1], [7, 6, 5, 4, 1], [4e-6, 3e-6, 2e-6, 1e-6, 0]], dtype=np.float32
        )  # type: ignore

        self.assertTrue(np.allclose(loaded_pc.points, expected_points))

    @unittest.mock.patch("nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points")
    def test_coordinate_transforms(self, prepare_pointcloud_points_mock: Mock) -> None:
        """
        Make sure points and timestamps accumulate properly with multiple sweeps.
        """
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(points=np.array([[100], [200], [300]], dtype=np.float32))
        mock_lidarpc_rec.prev.load.return_value = LidarPointCloud(points=np.array([[10], [20], [30]], dtype=np.float32))
        mock_lidarpc_rec.prev.prev.load.return_value = LidarPointCloud(
            points=np.array([[1], [2], [3]], dtype=np.float32)
        )

        mock_lidarpc_rec.timestamp = 507
        mock_lidarpc_rec.prev.timestamp = 504
        mock_lidarpc_rec.prev.prev.timestamp = 500

        mock_lidarpc_rec.lidar.trans_matrix = np.eye(4)
        mock_lidarpc_rec.lidar.trans_matrix_inv = np.eye(4)

        # Let's say the car points north here
        mock_lidarpc_rec.ego_pose.trans_matrix_inv = np.eye(4)
        # car points west, ego x-axis is global y-axis
        mock_lidarpc_rec.prev.ego_pose.trans_matrix = np.array(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        # car points east, ego x-axis is global negative y-axis
        mock_lidarpc_rec.prev.prev.ego_pose.trans_matrix = np.array(
            [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(
            nuplandb, token="abc", nsweeps=3, max_distance=1000, min_distance=0, sweep_map="sweep_idx"
        )
        expected_points = np.array(
            [[2, -20, 100], [-1, 10, 200], [3, 30, 300], [1, 2, 3]], dtype=np.float32
        )  # type: ignore

        self.assertTrue(np.allclose(loaded_pc.points, expected_points))


class TestLoadBoxes(unittest.TestCase):
    """Tests for get_boxes() and get_future_box_sequence()"""

    def setUp(self) -> None:
        """Set up the test case."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc()

        # The default NuPlanDB splitter.
        self.future_horizon_len_s = 1
        self.future_interval_s = 0.05

    def test_can_run_get_future_box_sequence(self) -> None:
        """Test get future box sequence."""
        get_future_box_sequence(
            lidar_pcs=[self.lidar_pc, self.lidar_pc.next],
            frame=Frame.VEHICLE,
            future_horizon_len_s=self.future_horizon_len_s,
            trans_matrix_ego=self.lidar_pc.ego_pose.trans_matrix_inv,
            future_interval_s=self.future_interval_s,
        )

    def test_pack_future_boxes(self) -> None:
        """Test pack future boxes."""
        track_token_2_box_sequence = get_future_box_sequence(
            lidar_pcs=[self.lidar_pc, self.lidar_pc.next],
            frame=Frame.VEHICLE,
            future_horizon_len_s=self.future_horizon_len_s,
            trans_matrix_ego=self.lidar_pc.ego_pose.trans_matrix_inv,
            future_interval_s=self.future_interval_s,
        )
        boxes_with_futures = pack_future_boxes(
            track_token_2_box_sequence=track_token_2_box_sequence,
            future_horizon_len_s=self.future_horizon_len_s,
            future_interval_s=self.future_interval_s,
        )
        for box in boxes_with_futures:
            for horizon_idx, horizon_s in enumerate(box.get_all_future_horizons_s()):
                future_center = box.get_future_center_at_horizon(horizon_s)
                future_orientation = box.get_future_orientation_at_horizon(horizon_s)
                self.assertTrue(box.track_token is not None)
                expected_future_box = track_token_2_box_sequence[box.track_token][horizon_idx + 1]
                if expected_future_box is None:
                    np.testing.assert_array_equal(future_center, [np.nan, np.nan, np.nan])
                    self.assertEqual(future_orientation, None)
                else:
                    np.testing.assert_array_equal(expected_future_box.center, future_center)
                    self.assertEqual(expected_future_box.orientation, future_orientation)

    def test_load_boxes_from_lidarpc(self) -> None:
        """Test load all boxes from a lidar pc."""
        boxes = load_boxes_from_lidarpc(
            self.db,
            self.lidar_pc,
            ["pedestrian", "vehicle"],
            False,
            80.04,
            self.future_horizon_len_s,
            self.future_interval_s,
            {"pedestrian": 0, "vehicle": 1},
        )

        self.assertSetEqual({"pedestrian", "vehicle"}, set(boxes.keys()))
        self.assertEqual(len(boxes["pedestrian"]), 70)
        self.assertEqual(len(boxes["vehicle"]), 29)


class TestGetFutureEgoTrajectory(unittest.TestCase):
    """Test getting future ego trajectory."""

    def setUp(self) -> None:
        """Set up test case."""
        self.lidar_pc = get_test_nuplan_lidarpc()

        self.future_lidarpc_recs: List[LidarPc] = [self.lidar_pc]
        while len(self.future_lidarpc_recs) < 200:
            self.future_lidarpc_recs.append(self.future_lidarpc_recs[-1].next)
        self.future_ego_poses = [rec.ego_pose for rec in self.future_lidarpc_recs]

    def test_get_future_ego_trajectory(self) -> None:
        """Test getting future ego trajectory."""
        future_ego_traj = get_future_ego_trajectory(self.lidar_pc, self.future_ego_poses, np.eye(4), 5.0, 0.5)
        self.assertEqual(future_ego_traj[0, 3], self.lidar_pc.ego_pose.timestamp)
        self.assertEqual(len(future_ego_traj), 11)
        self.assertLessEqual(abs((future_ego_traj[-1, 3] - future_ego_traj[0, 3]) / 1.0e6 - 5.0), 0.5)

    def test_get_future_ego_trajectory_not_enough(self) -> None:
        """Test getting future ego trajectory when there are not enough ego poses."""
        future_ego_traj = get_future_ego_trajectory(self.lidar_pc, self.future_ego_poses[:50], np.eye(4), 5.0, 0.5)
        self.assertEqual(future_ego_traj[0, 3], self.lidar_pc.ego_pose.timestamp)
        self.assertEqual(len(future_ego_traj), 11)
        np.testing.assert_equal(future_ego_traj[-1, :], [np.nan, np.nan, np.nan, np.nan])


class TestRenderOnMap(unittest.TestCase):
    """Test rendering on map."""

    def setUp(self) -> None:
        """Set up test case."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()

        self.future_lidarpc_recs: List[LidarPc] = [self.lidar_pc]
        while len(self.future_lidarpc_recs) < 200:
            self.future_lidarpc_recs.append(self.future_lidarpc_recs[-1].next)
        self.future_ego_poses = [rec.ego_pose for rec in self.future_lidarpc_recs]

    def test_render_on_map(self) -> None:
        """Test render on map."""
        render_on_map(
            self.lidar_pc,
            self.db,
            self.lidar_pc.boxes(),
            self.future_ego_poses,
            render_boxes_with_velocity=True,
            render_map_raster=False,
            render_vector_map=True,
            with_random_color=True,
            render_future_ego_poses=True,
        )


if __name__ == "__main__":
    unittest.main()
