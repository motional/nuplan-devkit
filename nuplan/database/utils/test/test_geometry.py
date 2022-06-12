import random
import unittest
from typing import List

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_almost_equal
from pyquaternion import Quaternion

from nuplan.database.utils.geometry import minimum_bounding_rectangle, quaternion_yaw, transform_matrix, view_points


class TestTransformMatrix(unittest.TestCase):
    """Test TransformMatrix."""

    def test_transform_matrix(self) -> None:
        """Test transform matrix using translation and rotation."""
        # unittest 1. generating translation matrix.
        zero_rotation = Quaternion(axis=(0.0, 0.0, 1.0), angle=0.0)

        for _ in range(100):
            x_trans = random.uniform(-100.0, 100.0)
            y_trans = random.uniform(-100.0, 100.0)
            z_trans = random.uniform(-100.0, 100.0)

            translation = np.array([x_trans, y_trans, z_trans])  # type: ignore
            tm = transform_matrix(translation, zero_rotation, False)

            tm_test = np.eye(4)
            tm_test[0:3, 3] = translation

            assert_array_almost_equal(tm, tm_test)

        # unittest 2. generating rotation matrix along x, y, and z axis.
        # rotation matrix inspired from https://en.wikipedia.org/wiki/Rotation_matrix

        zero_translation = np.array([0.0, 0.0, 0.0])  # type: ignore

        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        z_axis = (0.0, 0.0, 1.0)

        for axis_idx, axis in enumerate([x_axis, y_axis, z_axis]):
            for theta in np.linspace(-4.0 * np.pi, 4.0 * np.pi, 100):
                rotation = Quaternion(axis=axis, angle=theta)
                tm = transform_matrix(zero_translation, rotation, False)

                tm_test = np.eye(4)
                tm_test[(axis_idx + 1) % 3, (axis_idx + 1) % 3] = np.cos(theta)
                tm_test[(axis_idx + 1) % 3, (axis_idx + 2) % 3] = -np.sin(theta)
                tm_test[(axis_idx + 2) % 3, (axis_idx + 1) % 3] = np.sin(theta)
                tm_test[(axis_idx + 2) % 3, (axis_idx + 2) % 3] = np.cos(theta)

                assert_array_almost_equal(tm, tm_test)

        # unittest 3. generating transformation matrix for general values.

        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        z_axis = (0.0, 0.0, 1.0)

        for axis_idx, axis in enumerate([x_axis, y_axis, z_axis]):
            for theta in np.linspace(-4.0 * np.pi, 4.0 * np.pi, 100):
                x_trans = random.uniform(-100.0, 100.0)
                y_trans = random.uniform(-100.0, 100.0)
                z_trans = random.uniform(-100.0, 100.0)

                translation = np.array([x_trans, y_trans, z_trans])
                rotation = Quaternion(axis=axis, angle=theta)

                tm = transform_matrix(translation, rotation, False)

                tm_test = np.eye(4)
                tm_test[(axis_idx + 1) % 3, (axis_idx + 1) % 3] = np.cos(theta)
                tm_test[(axis_idx + 1) % 3, (axis_idx + 2) % 3] = -np.sin(theta)
                tm_test[(axis_idx + 2) % 3, (axis_idx + 1) % 3] = np.sin(theta)
                tm_test[(axis_idx + 2) % 3, (axis_idx + 2) % 3] = np.cos(theta)

                tm_test[0:3, 3] = translation

                assert_array_almost_equal(tm, tm_test)

        # unittest 4. comparison of inverse transformation matrix and algebraically calculated inverse matrix.

        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        z_axis = (0.0, 0.0, 1.0)

        for axis_idx, axis in enumerate([x_axis, y_axis, z_axis]):
            for theta in np.linspace(-4.0 * np.pi, 4.0 * np.pi, 100):
                x_trans = random.uniform(-100.0, 100.0)
                y_trans = random.uniform(-100.0, 100.0)
                z_trans = random.uniform(-100.0, 100.0)

                translation = np.array([x_trans, y_trans, z_trans])
                rotation = Quaternion(axis=axis, angle=theta)

                tm = transform_matrix(translation, rotation, False)
                inverse_tm = transform_matrix(translation, rotation, True)
                assert_array_almost_equal(inverse_tm, np.linalg.inv(tm))

        # unittest 5. commutativity test for translation matrix

        zero_rotation = Quaternion(axis=(0.0, 0.0, 1.0), angle=0.0)
        for _ in range(100):
            x_trans1 = random.uniform(-100.0, 100.0)
            y_trans1 = random.uniform(-100.0, 100.0)
            z_trans1 = random.uniform(-100.0, 100.0)

            translation1 = np.array([x_trans1, y_trans1, z_trans1])  # type: ignore
            tm1 = transform_matrix(translation1, zero_rotation, False)

            x_trans2 = random.uniform(-100.0, 100.0)
            y_trans2 = random.uniform(-100.0, 100.0)
            z_trans2 = random.uniform(-100.0, 100.0)

            translation2 = np.array([x_trans2, y_trans2, z_trans2])  # type: ignore
            tm2 = transform_matrix(translation2, zero_rotation, False)

            assert_array_almost_equal(tm1 * tm2, tm2 * tm1)

        # unittest 6. commutativity test for rotation matrix

        zero_translation = np.array([0.0, 0.0, 0.0])
        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        z_axis = (0.0, 0.0, 1.0)

        for _ in range(100):
            axis1 = random.choice([x_axis, y_axis, z_axis])
            theta1 = random.uniform(-4.0 * np.pi, 4.0 * np.pi)
            rotation1 = Quaternion(axis=axis1, angle=theta1)
            tm1 = transform_matrix(zero_translation, rotation1, False)

            axis2 = random.choice([x_axis, y_axis, z_axis])
            theta2 = random.uniform(-4.0 * np.pi, 4.0 * np.pi)
            rotation2 = Quaternion(axis=axis2, angle=theta2)
            tm2 = transform_matrix(zero_translation, rotation2, False)

            assert_array_almost_equal(tm1 * tm2, tm2 * tm1)


class TestViewPoints(unittest.TestCase):
    """Test ViewPoints."""

    def test_view_points(self) -> None:
        """Test expected value of view_points()."""
        # unittest 1. image origin and camera origin are identical.

        for _ in range(100):
            intrinsic = np.eye(3)
            focal = random.uniform(0.0, 10.0)
            intrinsic[0, 0] = focal
            intrinsic[1, 1] = focal

            pc1 = np.random.uniform(-100.0, 100.0, (3, 100))
            pc2: npt.NDArray[np.float64] = np.random.uniform(-100.0, 100.0) * pc1
            pc1_in_img = view_points(pc1, intrinsic, True)
            pc2_in_img = view_points(pc2, intrinsic, True)
            assert_array_almost_equal(pc1_in_img, pc2_in_img)

        # unittest 2. image origin and camera origin are not identical.

        for _ in range(100):
            intrinsic = np.eye(3)
            focal = random.uniform(0.0, 10.0)
            intrinsic[0, 0] = focal
            intrinsic[1, 1] = focal
            x_trans = random.uniform(-100.0, 100.0)
            y_trans = random.uniform(-100.0, 100.0)
            intrinsic[0, 2] = x_trans
            intrinsic[1, 2] = y_trans

            pc3 = np.random.uniform(-100.0, 100.0, (3, 100))
            pc4: npt.NDArray[np.float64] = np.random.uniform(-100.0, 100.0) * pc3
            pc3_in_img = view_points(pc3, intrinsic, True)
            pc4_in_img = view_points(pc4, intrinsic, True)
            assert_array_almost_equal(pc3_in_img, pc4_in_img)


class TestQuaternionYaw(unittest.TestCase):
    """Test QuaternionYaw."""

    def test_quaternion_yaw(self) -> None:
        """Test valid and invalid inputs for quaternion_yaw()."""
        # Misc yaws.
        for yaw_in in np.linspace(-10, 10, 100):
            q = Quaternion(axis=(0, 0, 1), angle=yaw_in)
            yaw_true = yaw_in % (2 * np.pi)
            if yaw_true > np.pi:
                yaw_true -= 2 * np.pi
            yaw_test = quaternion_yaw(q)
            self.assertAlmostEqual(yaw_true, yaw_test)

        # Non unit axis vector.
        yaw_in = np.pi / 4
        q = Quaternion(axis=(0, 0, 0.5), angle=yaw_in)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)

        # Inverted axis vector.
        yaw_in = np.pi / 4
        q = Quaternion(axis=(0, 0, -1), angle=yaw_in)
        yaw_test = -quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)

        # Rotate around another axis.
        yaw_in = np.pi / 4
        q = Quaternion(axis=(0, 1, 0), angle=yaw_in)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(0, yaw_test)

        # Rotate around two axes jointly.
        yaw_in = np.pi / 2
        q = Quaternion(axis=(0, 1, 1), angle=yaw_in)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)

        # Rotate around two axes separately.
        yaw_in = np.pi / 2
        q = Quaternion(axis=(0, 0, 1), angle=yaw_in) * Quaternion(axis=(0, 1, 0), angle=0.5821)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)


class TestMinimumBoundingRectangle(unittest.TestCase):
    """Tests for the minimum_bounding_rectangle() methods."""

    def check_minimum_bounding_rectangle(
        self, rect_points: npt.NDArray[np.float64], points_to_check: List[List[int]]
    ) -> None:
        """
        Given the points of the minimum rectangle and the points to check, this function checks whether each point
        in points_to_check lies in rect_points.
        :param rect_points: The points of the minimum rectangle.
        :param points_to_check: Points to check if they lie in the minimum rectangle.
        """
        # Assert that the shape is correct.
        self.assertTrue(rect_points.shape == (4, 2))

        # Round off the points in the returned rectangle. We round off the returned points for simplicity in testing.
        # np.equal() does not have a tolerance parameter.
        rect_points = np.around(rect_points, decimals=3)

        for point in points_to_check:
            self.assertTrue(np.equal(rect_points, np.around(point, decimals=3)).all(1).any())

    def test_all_square_vertices(self) -> None:
        """
        Use the vertices of a square as the input points. The minimum bounding rectangle for them would be the same
        square.
        """
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # type: ignore

        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_all_rectangle_vertices(self) -> None:
        """
        Use the vertices of a rectangle as the input points. The minimum bounding rectangle for them would be the
        complete rectangle.
        """
        points = np.array([[0, 0], [2, 1], [2, 0], [0, 1]])  # type: ignore

        rect_points = minimum_bounding_rectangle(points)

        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [2, 0], [0, 1], [2, 1]])

    def test_three_square_vertices(self) -> None:
        """
        Use the three vertices of a square as the input points. The minimum bounding rectangle for them would be the
        complete square.
        """
        points = np.array([[0, 0], [1, 1], [0, 1]])  # type: ignore

        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])

        points = np.array([[1, 0], [1, 1], [0, 1]])

        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])

        points = np.array([[0, 0], [1, 1], [1, 0]])

        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_lots_of_random_points_in_a_square(self) -> None:
        """
        Use the three vertices of a square as the input points. Then concatenate a bunch of random points inside the
        square to those points. The minimum bounding rectangle for them would be the original square.
        """
        points = np.array([[0, 0], [1, 1], [0, 1]])  # type: ignore

        # Now append a bunch of random points inside the same square to the three vertices.
        pts_inside_square = np.random.rand(30, 2)
        points = np.concatenate([points, pts_inside_square])

        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_lots_of_random_points_in_a_rotated_square(self) -> None:
        """
        Use the four vertices of a square as the input points. Then concatenate a bunch of random points inside the
        square to those points. Finally rotate all the points by a fixed angle. The minimum bounding rectangle for them
        would be the original square rotated by the same angle chosen in the last step.
        """
        points = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])  # type: ignore

        # Now append a bunch of random points inside the same square to the three vertices.
        pts_inside_square = np.random.rand(30, 2)
        points = np.concatenate([points, pts_inside_square])

        # Rotation matrix
        rand_angle = np.random.randn()
        rot_mat = np.array(
            [[np.cos(rand_angle), np.sin(rand_angle)], [-np.sin(rand_angle), np.cos(rand_angle)]]
        )  # type: ignore

        rect_points = minimum_bounding_rectangle(np.dot(rot_mat, points.T).T)

        self.check_minimum_bounding_rectangle(
            rect_points, np.dot(rot_mat, np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T).T
        )


if __name__ == '__main__':
    unittest.main()
