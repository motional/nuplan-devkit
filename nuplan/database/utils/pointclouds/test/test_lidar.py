import struct
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from pyquaternion import Quaternion

from nuplan.database.utils.pointclouds.lidar import PCD_TIMESTAMP_FIELD_NAME, LidarPointCloud


class TestPointCloud(unittest.TestCase):
    """Test Class for Point Cloud."""

    def test_load_pcd_bin_v1(self) -> None:
        """Testing if points in binary format v1 can be read."""
        pcd_expected = np.array(
            [
                [3.5999999, -3.0999999, 0, 1, 0.5, -1],
                [1.0, -3.01, 10.0, 0.4, 10, -1],
                [4.5999999, -2.90001, -1.0, 0.1, 1.5, -1],
            ],
            dtype=np.float32,
        )  # type: ignore

        file_path = tempfile.NamedTemporaryFile()

        with open(file_path.name, 'w+b'):
            for point in pcd_expected:
                file_path.write(struct.pack("5f", point[0], point[1], point[2], point[3], point[4]))
            _ = file_path.seek(0)
            pcd = LidarPointCloud.load_pcd_bin(file_path.name)
            assert np.all(pcd == pcd_expected.T)

    def test_load_pcd_bin_v2(self) -> None:
        """Testing if points in binary format v2 can be read."""
        pcd_expected = np.array(
            [
                [3.5999999, -3.0999999, 0, 1, 0.5, -1],
                [1.0, -3.01, 10.0, 0.4, 10, -1],
                [4.5999999, -2.90001, -1.0, 0.1, 1.5, -1],
            ],
            dtype=np.float32,
        )  # type: ignore

        file_path = tempfile.NamedTemporaryFile()

        with open(file_path.name, 'w+b'):
            for point in pcd_expected:
                file_path.write(struct.pack("6f", point[0], point[1], point[2], point[3], point[4], point[5]))
            _ = file_path.seek(0)
            pcd = LidarPointCloud.load_pcd_bin(file_path.name, 2)
            assert np.all(pcd == pcd_expected.T)

    def test_nbr_points(self) -> None:
        """Testing if the number of points in the pointcloud is returned."""
        test_pointcloud = np.array(
            [
                [35, 35, 0, 0, 0],
                [20.0, 30.0, 2000, 0, 0],
                [30.0, 20.0, 0, 0, 0],
                [8.0, 8.0, 0, 0, 0],
                [0.0, 15.0, 10, 0, 0],
            ]
        )  # type: ignore

        pc = LidarPointCloud(test_pointcloud.T)
        self.assertEqual(pc.nbr_points(), 5)

    def test_subsample(self) -> None:
        """Testing if the correct number of points are sampled given the ratio."""
        test_pointcloud = np.zeros((100, 5))

        pc = LidarPointCloud(test_pointcloud.T)
        pc.subsample(ratio=0.5)
        self.assertEqual(pc.nbr_points(), 50)

        pc.subsample(ratio=0.2)
        self.assertEqual(pc.nbr_points(), 10)

        # number of points are floored
        pc.subsample(ratio=0.18)
        self.assertEqual(pc.nbr_points(), 1)

    def test_1d_array_input(self) -> None:
        """Testing if can do translate/rotate function from single point input array."""
        pc = LidarPointCloud(np.array([0, 0, 0, 0, 0]))
        test_translate = np.array([0, 0, 1])  # type: ignore
        pc.translate(test_translate)
        assert_array_equal(pc.points[:, 0], np.array([0, 0, 1, 0, 0]))

        theta = np.pi
        test_rot_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]]
        )  # type: ignore
        pc.rotate(Quaternion(matrix=test_rot_matrix))

        self.assertAlmostEqual(pc.points[0, 0], 0)
        self.assertAlmostEqual(pc.points[1, 0], 0)
        self.assertAlmostEqual(pc.points[2, 0], -1)

    def test_remove_close(self) -> None:
        """Testing if points within a certain radius from origin (in bird view) are correctly removed."""
        test_pointcloud = np.array(
            [
                [35, 35, 0, 0, 0],
                [20.0, 30.0, 2000, 0, 0],
                [30.0, 20.0, 0, 0, 0],
                [8.0, 8.0, 0, 0, 0],
                [0.0, 15.0, 10, 0, 0],
            ]
        )  # type: ignore

        pc = LidarPointCloud(test_pointcloud.T)
        pc.remove_close(5)
        self.assertEqual(pc.nbr_points(), 5)

        pc.remove_close(12)
        self.assertEqual(pc.nbr_points(), 4)

        # if border, keep.
        pc.remove_close(15)
        self.assertEqual(pc.nbr_points(), 4)

        pc.remove_close(36.1)
        self.assertEqual(pc.nbr_points(), 1)

    def test_radius_filter(self) -> None:
        """Testing if points within a certain radius from origin (in bird view) is correctly removed."""
        test_pointcloud = np.array(
            [
                [35, 35, 0, 0, 0],
                [20.0, 30.0, 2000, 0, 0],
                [30.0, 20.0, 0, 0, 0],
                [8.0, 8.0, 0, 0, 0],
                [0.0, 15.0, 10, 0, 0],
            ]
        )  # type: ignore

        pointcloud = LidarPointCloud(test_pointcloud.T)

        pc = pointcloud.copy()
        pc.radius_filter(5)
        self.assertEqual(pc.nbr_points(), 0)

        pc = pointcloud.copy()
        pc.radius_filter(12)
        self.assertEqual(pc.nbr_points(), 1)

        # if at border, keep.
        pc = pointcloud.copy()
        pc.radius_filter(15)
        self.assertEqual(pc.nbr_points(), 2)

        pc = pointcloud.copy()
        pc.radius_filter(36.1)
        self.assertEqual(pc.nbr_points(), 4)

    def test_scale(self) -> None:
        """Testing if the lidar xyz coordinates are scaled."""
        test_pointcloud = np.array(
            [
                [35, 35, 0, 0, 0],
                [20.0, 30.0, 2000, 0, 0],
                [30.0, 20.0, 0, 0, 0],
                [8.0, 8.0, 0, 0, 0],
                [0.0, 15.0, 10, 0, 0],
            ]
        )  # type: ignore

        test_pc = test_pointcloud.copy()
        pc = LidarPointCloud(test_pc.T)
        pc.scale((2, 2, 2))
        test_pc_scaled = test_pointcloud.copy()
        test_pc_scaled[:, 0:3] *= 2
        pc_scaled = LidarPointCloud(test_pc_scaled.T)
        self.assertEqual(pc, pc_scaled)

    def test_translate_simple(self) -> None:
        """Testing if points are translated correctly given a translate vector."""
        pc = LidarPointCloud(np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0, 0.0]]).T)

        test_translate = np.array([5.2, 10.4, 15.1])  # type: ignore
        pc.translate(test_translate)

        assert_array_equal(pc.points[:, 0], np.array([5.2, 10.4, 15.1, 0, 0]))
        assert_array_equal(pc.points[:, 1], np.array([6.2, 12.4, 18.1, 0, 0]))

    def test_rotate_simple(self) -> None:
        """Testing if points are rotated correctly given a rotation matrix."""
        # rotation of x by one quadrant, will be in the yz plane
        theta = np.pi / 4
        test_rot_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]]
        )  # type: ignore

        pc = LidarPointCloud(np.array([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]).T)
        pc.rotate(Quaternion(matrix=test_rot_matrix))

        self.assertAlmostEqual(pc.points[0, 0], 0)
        self.assertAlmostEqual(pc.points[1, 0], -1 / np.sqrt(2))
        self.assertAlmostEqual(pc.points[2, 0], 1 / np.sqrt(2))

    def test_copy(self) -> None:
        """Verify that copy works as expected."""
        pc_orig = LidarPointCloud.make_random()
        pc_copy = pc_orig.copy()

        # Confirm that points are equivalent.
        self.assertEqual(pc_orig, pc_copy)

        # Check that copy are independent after changes to original.
        pc_orig.points[0, 0] += 1
        self.assertNotEqual(pc_orig, pc_copy)

    def test_read_pcd_ascii_xyz(self) -> None:
        """Test making a LidarPointCloud with x, y, and z fields from a .pcd file with ascii data."""
        pcd_contents = b"""#.PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
3.5999999 -3.0999999 0
1.0 -3.01 10.0
4.5999999 -2.90001 -1.0"""

        temp_file = tempfile.NamedTemporaryFile(suffix=".pcd")
        temp_file.write(pcd_contents)
        _ = temp_file.seek(0)

        pcd = LidarPointCloud.from_file(temp_file.name)
        self.assertEqual(pcd.nbr_points(), 3)
        # Zeros will be filled for intensity
        expected_points = np.array(
            [[3.5999999, -3.0999999, 0, 0], [1.0, -3.01, 10, 0], [4.5999999, -2.90001, -1.0, 0]]
        ).T  # type: ignore
        self.assertEqual(np.all(np.isclose(pcd.points, expected_points)), True)

    def test_read_pcd_ascii_xyzi(self) -> None:
        """Test making a LidarPointCloud with x, y, z, and intensity fields from a .pcd file with ascii data."""
        pcd_contents = b"""#.PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z r intensity rcs
SIZE 4 4 4 4 4 4
TYPE F F F F F F
COUNT 1 1 1 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
3.5999999 -3.0999999 0 1 0.5 7.5
1.0 -3.01 10.0 0.4 10 2.5
4.5999999 -2.90001 -1.0 0.1 1.5 -3.5"""

        temp_file = tempfile.NamedTemporaryFile(suffix=".pcd")
        temp_file.write(pcd_contents)
        _ = temp_file.seek(0)

        pcd = LidarPointCloud.from_file(temp_file.name)
        self.assertEqual(pcd.nbr_points(), 3)
        expected_points = np.array(
            [[3.5999999, -3.0999999, 0, 0.5], [1.0, -3.01, 10, 10], [4.5999999, -2.90001, -1.0, 1.5]]
        ).T  # type: ignore
        self.assertEqual(np.all(np.isclose(pcd.points, expected_points)), True)

    def test_read_pcd_ascii_xyzit(self) -> None:
        """Test making a LidarPointCloud with x, y, z, intensity, and time fields from a .pcd file with ascii data."""
        pcd_contents = f"""#.PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z r intensity rcs {PCD_TIMESTAMP_FIELD_NAME}
SIZE 4 4 4 4 4 4 4
TYPE F F F F F F F
COUNT 1 1 1 1 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
3.5999999 -3.0999999 0 1 0.5 7.5 0
1.0 -3.01 10.0 0.4 10 2.5 0.05
4.5999999 -2.90001 -1.0 0.1 1.5 -3.5 0.1""".encode(
            'utf-8'
        )

        temp_file = tempfile.NamedTemporaryFile(suffix=".pcd")
        temp_file.write(pcd_contents)
        _ = temp_file.seek(0)

        pcd = LidarPointCloud.from_file(temp_file.name)
        self.assertEqual(pcd.nbr_points(), 3)
        expected_points = np.array(
            [[3.5999999, -3.0999999, 0, 0.5, 0], [1.0, -3.01, 10, 10, 0.05], [4.5999999, -2.90001, -1.0, 1.5, 0.1]]
        ).T  # type: ignore
        self.assertEqual(np.all(np.isclose(pcd.points, expected_points)), True)

    def test_read_pcd_ascii_shuffled_field_order(self) -> None:
        """
        Test making a LidarPointCloud with x, y, z, intensity, and time fields from a .pcd file
        with ascii data where the fields are in an unusual order.
        """
        pcd_contents = f"""#.PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS {PCD_TIMESTAMP_FIELD_NAME} intensity r rcs x y z
SIZE 4 4 4 4 4 4 4
TYPE F F F F F F F
COUNT 1 1 1 1 1 1 1
WIDTH 2
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 2
DATA ascii
1 2 3 4 5 6 7
8 9 10 11 12 13 14""".encode(
            'utf-8'
        )

        temp_file = tempfile.NamedTemporaryFile(suffix=".pcd")
        temp_file.write(pcd_contents)
        _ = temp_file.seek(0)

        pcd = LidarPointCloud.from_file(temp_file.name)
        self.assertEqual(pcd.nbr_points(), 2)
        # We always return the result in "xyzit" order
        expected_points = np.array([[5, 6, 7, 2, 1], [12, 13, 14, 9, 8]]).T  # type: ignore
        self.assertEqual(np.all(np.isclose(pcd.points, expected_points)), True)

    def test_range_filter(self) -> None:
        """Test if Range filter works as expected."""
        points_orig = np.array(
            [
                [2.26, -0.76, 4.72, -5.46, 9.54, -8.89, 5.45, 7.05, -0.89, 8.58],
                [-0.88, 1.81, -9.12, 3.32, 3.13, -8.67, -5.11, 6.22, 9.39, -3.25],
                [4.42, -9.08, 0.12, 2.5, -4.23, 2.08, 8.12, 9.22, -8.71, 3.9],
                [2.25, 4.32, 4.53, 2.88, 2.84, 0.79, 7.62, 1.21, 3.3, 0.52],
                [9.72, 9.43, 3.67, 9.99, 5.56, 3.15, 0.02, 7.07, 8.64, 6.16],
            ],
            dtype=float,
        )  # type: ignore

        pc = LidarPointCloud(points_orig)
        # Filter along x.
        pc.range_filter(xrange=(-2, 2))
        should_match = np.array([[-0.76, 1.81, -9.08, 4.32, 9.43], [-0.89, 9.39, -8.71, 3.3, 8.64]]).T  # type: ignore
        self.assertTrue(np.array_equal(pc.points, should_match))

        # Filter along all axes.
        pc = LidarPointCloud(points_orig)
        pc.range_filter(xrange=(5, 10), yrange=(-5, 0), zrange=(3, 5))
        should_match = np.array([[8.58, -3.25, 3.9, 0.52, 6.16]]).T
        self.assertTrue(np.array_equal(pc.points, should_match))

        # No points fall in the range.
        pc = LidarPointCloud(points_orig)
        pc.range_filter(xrange=(1000, 2000))
        self.assertEqual(pc.nbr_points(), 0)

        # Matches all points.
        pc = LidarPointCloud(points_orig)
        pc.range_filter(xrange=(-100, 100), yrange=(-100, 100), zrange=(-100, 100))
        self.assertTrue(np.array_equal(pc.points, points_orig))

    def test_transform(self) -> None:
        """
        Test the transform function (example transformation matrices taken from
        https://www.springer.com/cda/content/document/cda_downloaddocument/9789048137756-c2.pdf?SGWID=0-0-45-1123955-p173940737
        """
        test_points = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [0.0, 0.0, 0.0]])  # type: ignore
        pc = LidarPointCloud(test_points.copy())

        # translate along x and y direc.
        pc.transform(np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0.0, 0.0, 0.0, 1]]))
        shouldMatch = np.array([[2, 1, 2], [1, 2, 2], [0, 0, 1], [0.0, 0.0, 0.0]])  # type: ignore
        self.assertTrue(np.array_equal(pc.points, shouldMatch))

        # Rotate about z 90deg then y 90deg then translate x by 4, y by -3, and z by 7
        pc = LidarPointCloud(test_points.copy())
        pc.transform(np.array([[0, 0, 1, 4], [1, 0, 0, -3], [0, 1, 0, 7], [0.0, 0.0, 0.0, 1]]))
        shouldMatch = np.array([[4, 4, 5], [-2, -3, -2], [7, 8, 8], [0.0, 0.0, 0.0]])
        self.assertTrue(np.array_equal(pc.points, shouldMatch))

    def test_equality(self) -> None:
        """Test equality of two points cloud based on element-wise difference."""
        # Test an equal case even if small difference.
        test_points = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [0.0, 0.0, 0.0]])  # type: ignore
        pc = LidarPointCloud(test_points.copy())
        test_points_2 = np.asarray(
            [[1.0000001, 0.0000001, 1], [0.0000001, 1.0000001, 1], [0, 0.0, 1], [0.0, 0.0, 0.0]]
        )  # type: ignore

        pc2 = LidarPointCloud(test_points_2.copy())
        self.assertEqual(pc, pc2)

        # Test an unequal case with two random PointClouds.
        pc = LidarPointCloud.make_random()
        pc2 = LidarPointCloud.make_random()
        self.assertNotEqual(pc, pc2)

    def test_rotate_composite(self) -> None:
        """Testing if points are rotated correctly for a composite rotation sequence."""
        # Two unit points with all zero features.
        test_point = np.array([[0, 0, -1, 0, 0], [0, -1, 0, 0, 0]]).T  # type: ignore
        # Angle for rotation about x, y and z axes.
        alpha, beta, gamma = np.pi, np.pi / 2, np.pi / 2

        test_rot_matrix_alpha = np.array(
            [[1.0, 0.0, 0.0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]]
        )  # type: ignore

        test_rot_matrix_beta = np.array(
            [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
        )  # type: ignore

        test_rot_matrix_gamma = np.array(
            [[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]
        )  # type: ignore

        # Expected result.
        rotated_test_point = np.array([[0, 1, 0, 0, 0], [-1, 0, 0, 0, 0]]).T  # type: ignore

        pc = LidarPointCloud(test_point)
        pc.rotate(Quaternion(matrix=test_rot_matrix_alpha))
        pc.rotate(Quaternion(matrix=test_rot_matrix_beta))
        pc.rotate(Quaternion(matrix=test_rot_matrix_gamma))

        # Compare both arrays.
        assert_array_equal(pc.points, rotated_test_point)


if __name__ == '__main__':
    unittest.main()
