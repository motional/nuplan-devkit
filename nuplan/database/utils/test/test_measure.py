import math
import unittest
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_almost_equal
from pyquaternion import Quaternion

from nuplan.database.utils import measure
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.geometry import quaternion_yaw


class TestAngleDiff(unittest.TestCase):
    """Unittests for angle difference."""

    def test_angle_diff_2pi(self) -> None:
        """Tests angle diff function for 2 pi."""
        period = 2 * math.pi

        x, y = math.pi, math.pi
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)

        x, y = math.pi, -math.pi
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)

        x, y = -math.pi / 6, math.pi / 6
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -math.pi / 3)

        x, y = 2 * math.pi / 3, -2 * math.pi / 3
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -2 * math.pi / 3)

        x, y = 8 * math.pi / 3, -2 * math.pi / 3
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -2 * math.pi / 3)

        x, y = 0, math.pi
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -math.pi)

    def test_angle_diff_pi(self) -> None:
        """Tests angle diff function for pi."""
        period = math.pi

        x, y = math.pi, math.pi
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)

        x, y = math.pi, -math.pi
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)

        x, y = -math.pi / 6, math.pi / 6
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -math.pi / 3)

        x, y = 2 * math.pi / 3, -2 * math.pi / 3
        self.assertAlmostEqual(measure.angle_diff(x, y, period), math.pi / 3)

        x, y = 8 * math.pi / 3, -2 * math.pi / 3
        self.assertAlmostEqual(measure.angle_diff(x, y, period), math.pi / 3)

        x, y = 0, math.pi
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)

    def test_quaternion(self) -> None:
        """Tests the angle difference between two yaw angles from two quaternions."""
        x = quaternion_yaw(Quaternion(axis=(0, 0, 1), angle=1.1 * np.pi))
        y = quaternion_yaw(Quaternion(axis=(0, 0, 1), angle=0.9 * np.pi))
        diff = measure.angle_diff(x, y, period=2 * np.pi)

        self.assertAlmostEqual(diff, 0.2 * np.pi)


class TestAngleDiffNumpy(unittest.TestCase):
    """Unittests for angle difference of numpy array."""

    def test_zero_shape(self) -> None:
        """Tests with zero input."""
        period = 2 * np.pi

        x = np.zeros((0,))
        y = np.zeros((0,))
        answer = np.zeros((0,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

    def test_angle_diff_2pi(self) -> None:
        """Tests angle diff function for 2 pi."""
        period = 2 * np.pi

        x: npt.NDArray[np.float64] = np.pi * np.ones((2,))
        y: npt.NDArray[np.float64] = np.pi * np.ones((2,))
        answer = np.zeros((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = np.pi * np.ones((2,))
        y = -np.pi * np.ones((2,))
        answer = np.zeros((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = -np.pi / 6 * np.ones((2,))
        y = np.pi / 6 * np.ones((2,))
        answer = -np.pi / 3 * np.ones((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = 2 * np.pi / 3 * np.ones((2,))
        y = -2 * np.pi / 3 * np.ones((2,))
        answer = -2 * np.pi / 3 * np.ones((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = 8 * np.pi / 3 * np.ones((2,))
        y = -2 * np.pi / 3 * np.ones((2,))
        answer = -2 * np.pi / 3 * np.ones((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = np.zeros((2,))
        y = np.pi * np.ones((2,))
        answer = -np.pi * np.ones((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

    def test_angle_diff_pi(self) -> None:
        """Tests angle diff function for pi."""
        period = math.pi

        x: npt.NDArray[np.float64] = np.pi * np.ones((2,))
        y: npt.NDArray[np.float64] = np.pi * np.ones((2,))
        answer = np.zeros((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = np.pi * np.ones((2,))
        y = -np.pi * np.ones((2,))
        answer = np.zeros((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = -np.pi / 6 * np.ones((2,))
        y = np.pi / 6 * np.ones((2,))
        answer = -np.pi / 3 * np.ones((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = 2 * np.pi / 3 * np.ones((2,))
        y = -2 * np.pi / 3 * np.ones((2,))
        answer = np.pi / 3 * np.ones((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = 8 * np.pi / 3 * np.ones((2,))
        y = -2 * np.pi / 3 * np.ones((2,))
        answer = np.pi / 3 * np.ones((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)

        x = np.zeros((2,))
        y = np.pi * np.ones((2,))
        answer = np.zeros((2,))
        assert_almost_equal(measure.angle_diff_numpy(x, y, period), answer)


class TestBirdviewCenterDistanceBox(unittest.TestCase):
    """Unit test for birdview center distance."""

    def test_birdview_center_distance(self) -> None:
        """Test the l2 distance between birdview bounding box centers."""
        dist = measure.birdview_center_distance((0.0, 0.0, 1.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(dist, 0)
        dist = measure.birdview_center_distance((0.0, 0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(dist, 1)
        dist = measure.birdview_center_distance((0.0, 0.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0))
        self.assertAlmostEqual(dist, 1.4142135623730951)

    def test_birdview_center_distance_box(self) -> None:
        """Test the l2 distance between birdview bounding box centers in Box3D class format."""
        dist = measure.birdview_center_distance_box(
            Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0))
        )
        self.assertEqual(dist, 0)
        dist = measure.birdview_center_distance_box(
            Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0))
        )
        self.assertEqual(dist, 1)
        dist = measure.birdview_center_distance_box(
            Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 1, 0), (1, 1, 1), Quaternion(0, 0, 0, 0))
        )
        self.assertAlmostEqual(dist, 1.4142135623730951)

        # compare complicated for test_birdview_center_distance_box and birdview_center_distance
        dist1 = measure.birdview_center_distance_box(
            Box3D((4, 5, 0), (2, 2, 1), Quaternion(0, 0, 0, 0)),
            Box3D((1, 4, 0), (2, 4, 1), Quaternion(axis=(0, 0, 1), angle=np.pi / 3)),
        )
        dist2 = measure.birdview_center_distance((4.0, 5.0, 2.0, 2.0, 0.0), (1.0, 4.0, 2.0, 4.0, np.pi / 3.0))
        self.assertEqual(dist1, dist2)


class TestHausdorffDistance(unittest.TestCase):
    """Unit test for hausdorff_distance"""

    def test_hausdorff_distance(self) -> None:
        """Test Hausdorff distance between two 2d-boxes"""
        dist = measure.hausdorff_distance((0.0, 0.0, 1.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(dist, 0)

        dist = measure.hausdorff_distance((0.0, 0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(dist, 1.0)

        dist = measure.hausdorff_distance((0.0, 0.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0))
        self.assertAlmostEqual(dist, 1.4142135623730951)

        # 2 boxes of same center and width=1, length=2 but one is rotated.
        dist = measure.hausdorff_distance((1.0, 1.0, 1.0, 2.0, 0.0), (1.0, 1.0, 1.0, 2.0, np.pi / 2.0))
        self.assertAlmostEqual(dist, 0.5)

        # 2 boxes of same size, one is overlapping by w/2.
        dist = measure.hausdorff_distance((1.0, 1.0, 1.0, 2.0, 0.0), (1.0, 1.5, 1.0, 2.0, 0.0))
        self.assertAlmostEqual(dist, 0.5)

        # box1 is inside box2. box2 is double the size of box1.box1 and box2 is aligned in x axis.
        dist = measure.hausdorff_distance((1.0, 1.0, 1.0, 2.0, 0.0), (1.0, 2.0, 2.0, 4.0, 0.0))
        self.assertAlmostEqual(dist, np.sqrt(0.5**2 + 2**2))

        # box1 and box2 are square box. one box rotated by pi/4. The drawing would look like an open envelope.
        dist = measure.hausdorff_distance(
            (0.0, 0.0, 2.0 / np.sqrt(2.0), 2.0 / np.sqrt(2.0), 0.0), (0.0, 1.0 / np.sqrt(2.0), 1.0, 1.0, np.pi / 4.0)
        )
        self.assertAlmostEqual(dist, 1)

    def test_hausdorff_distance_box(self) -> None:
        """Test Hausdorff distance between two 2d-boxes in Box3D class."""
        dist = measure.hausdorff_distance_box(
            Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((0, 0, 0), (1, 1, 10), Quaternion(0, 0, 0, 0))
        )
        self.assertEqual(dist, 0)

        dist = measure.hausdorff_distance_box(
            Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0))
        )
        self.assertEqual(dist, 1.0)

        dist = measure.hausdorff_distance_box(
            Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 1, 0), (1, 1, 1), Quaternion(0, 0, 0, 0))
        )
        self.assertAlmostEqual(dist, 1.4142135623730951)

        # compare complicated for test_hausdorff_distance_box and hausdorff_distance
        dist1 = measure.hausdorff_distance_box(
            Box3D((4, 5, 0), (2, 2, 1), Quaternion(0, 0, 0, 0)),
            Box3D((1, 4, 0), (2, 4, 1), Quaternion(axis=(0, 0, 1), angle=np.pi / 3)),
        )
        dist2 = measure.hausdorff_distance((4.0, 5.0, 2.0, 2.0, 0.0), (1.0, 4.0, 2.0, 4.0, np.pi / 3.0))
        self.assertEqual(dist1, dist2)


class TestPseudoIOU(unittest.TestCase):
    """Test the birdview_pseudo_iou metric."""

    def test_pseudo_distance_2pi(self) -> None:
        """Test ad-hoc birdview distance of two 2-d boxes with period of 2 pi."""
        period = 2 * np.pi

        # Perfect match
        a = (0.0, 0.0, 1.0, 2.0, 0.0)
        b = (0.0, 0.0, 1.0, 2.0, 0.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)

        # Perfect match
        a = (0.0, 0.0, 1.0, 2.0, 0.0)
        b = (0.0, 0.0, 1.0, 2.0, 2.0 * math.pi)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)

        # Perfect box, large angle error
        a = (-10.0, 10.0, 0.1, 20.0, 0.0)
        b = (-10.0, 10.0, 0.1, 20.0, math.pi / 2.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), (math.pi / 2) / 5)

        # Perfect box, large angle error
        a = (-10, 10, 0.1, 20, 0)
        b = (-10, 10, 0.1, 20, math.pi)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), math.pi / 5)

        # Bad box, perfect angle
        a = (-100.0, -100.0, 100.0, 100.0, 0.0)
        b = (0.0, 0.0, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 398 / 5)

        # Bad match overall
        a = (-100.0, -100.0, 100.0, 100.0, 0.0)
        b = (0.0, 0.0, 1.0, 1.0, math.pi / 2.0)
        self.assertAlmostEqual(
            measure.birdview_corner_angle_mean_distance(a, b, period=period), (398 + math.pi / 2) / 5
        )

    def test_pseudo_distance_pi(self) -> None:
        """Test ad-hoc birdview distance of two 2-d boxes with period of pi."""
        period = np.pi

        # Perfect match
        a = (0.0, 0.0, 1.0, 2.0, 0.0)
        b = (0.0, 0.0, 1.0, 2.0, 0.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)

        # Perfect match
        a = (0.0, 0.0, 1.0, 2.0, 0.0)
        b = (0.0, 0.0, 1.0, 2.0, 2.0 * math.pi)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)

        # Perfect box, large angle error
        a = (-10.0, 10.0, 0.1, 20.0, 0.0)
        b = (-10.0, 10.0, 0.1, 20.0, math.pi / 2.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), (math.pi / 2) / 5)

        # Perfect box, perfect angle due to period
        a = (-10.0, 10.0, 0.1, 20.0, 0.0)
        b = (-10.0, 10.0, 0.1, 20.0, math.pi)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)

        # Bad box, perfect angle
        a = (-100.0, -100.0, 100.0, 100.0, 0.0)
        b = (0.0, 0.0, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 398 / 5)

        # Bad match overall
        a = (-100.0, -100.0, 100.0, 100.0, 0.0)
        b = (0.0, 0.0, 1.0, 1.0, math.pi / 2.0)
        self.assertAlmostEqual(
            measure.birdview_corner_angle_mean_distance(a, b, period=period), (398 + math.pi / 2) / 5
        )

    def test_pseudo_distance_box_pi(self) -> None:
        """Unit test for calculating ad-hoc birdview distance of two Box3D instances with period of pi."""
        period = np.pi

        # Perfect match
        a = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)

        # Perfect match
        a = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=2 * math.pi))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)

        # Perfect box, large angle error
        a = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi / 2))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), (math.pi / 2) / 5)

        # Perfect box, perfect angle due to period
        a = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)

        # Bad box, perfect angle
        a = Box3D(center=(-100, -100, 0), size=(100, 100, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 1, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 398 / 5)

        # Bad match overall
        a = Box3D(center=(-100, -100, 0), size=(100, 100, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 1, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi / 2))
        self.assertAlmostEqual(
            measure.birdview_corner_angle_mean_distance_box(a, b, period=period), (398 + math.pi / 2) / 5
        )

    def test_pseudo_distance_box_2pi(self) -> None:
        """Unit test for calculating ad-hoc birdview distance of two Box3D instances with period of 2 * pi."""
        period = 2 * np.pi

        # Perfect match
        a = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)

        # Perfect match
        a = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=2 * math.pi))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)

        # Perfect box, large angle error
        a = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi / 2))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), (math.pi / 2) / 5)

        # Perfect box, large angle error
        a = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), math.pi / 5)

        # Bad box, perfect angle
        a = Box3D(center=(-100, -100, 0), size=(100, 100, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 1, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 398 / 5)

        # Bad match overall
        a = Box3D(center=(-100, -100, 0), size=(100, 100, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 1, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi / 2))
        self.assertAlmostEqual(
            measure.birdview_corner_angle_mean_distance_box(a, b, period=period), (398 + math.pi / 2) / 5
        )


class TestAssign(unittest.TestCase):
    """Test hungarian algorithm in assign."""

    def test_assign_linear(self) -> None:
        """Test linear cost function."""
        gtboxes = [1, 2, 5]
        estboxes = [4]

        def distance_fcn(a: Any, b: Any) -> float:
            """
            Distance function.
            :param a: Input a.
            :param b: Input b.
            :return: distance between input.
            """
            return float(np.abs(a - b))

        # Test one successful match.
        pairs_index = measure.assign(gtboxes, estboxes, distance_fcn, 1.5)
        pairs = [(gtboxes[pair[0]], estboxes[pair[1]]) for pair in pairs_index]
        self.assertEqual(len(pairs), 1)
        self.assertTrue(5 in pairs[0])
        self.assertTrue(4 in pairs[0])

        # Test no matches
        pairs_index = measure.assign(gtboxes, estboxes, distance_fcn, 0.5)
        pairs = [(gtboxes[pair[0]], estboxes[pair[1]]) for pair in pairs_index]
        self.assertEqual(len(pairs), 0)

    def test_center_distance(self) -> None:
        """Test center distance cost function and new Hungarian algorithm variant."""
        distance_fcn = measure.birdview_center_distance

        # Replicating a scenario where original Hungarian algorithm assigns a FP prediction (-5,0) to a GT box (0, 0)
        gtboxes = [(0, 0), (5, 0)]
        estboxes = [(-5, 0), (0, 0.5)]

        matching = np.array(measure.assign(gtboxes, estboxes, distance_fcn, 2))  # type: ignore

        # GT box (0, 0) and predicted box (0, 0.5) will closely overlap and should be matched
        self.assertTrue((matching == [(0, 1)]).all())


class TestIOU(unittest.TestCase):
    """Test IOU related functions."""

    def test_intersection(self) -> None:
        """Test intersection of boxes."""
        a = (0.0, 0.0, 100.0, 100.0)

        # perfect intersection
        b = (0.0, 0.0, 100.0, 100.0)
        self.assertEqual(measure.intersection(a, b), 10000.0)

        # no intersection with box of no size
        b = (100.0, 100.0, 100.0, 100.0)
        self.assertEqual(measure.intersection(a, b), 0.0)

        # no intersection
        b = (100.0, 100.0, 200.0, 200.0)
        self.assertEqual(measure.intersection(a, b), 0.0)

        # Partial Intersection
        # dx = 100 (a) - 50 (b) = 50
        # dy = 100 (a) - 50 (b) = 50
        # Area = 50 * 50 = 2500
        b = (50.0, 50.0, 150.0, 150.0)
        self.assertEqual(measure.intersection(a, b), 2500.0)

    def test_union(self) -> None:
        """Test union of boxes."""
        a = (0.0, 0.0, 100.0, 100.0)

        # Two perfectly overlapping boxes.
        b = (0.0, 0.0, 100.0, 100.0)
        self.assertEqual(measure.union(a, b), 10000.0)

        # Test box with no size.
        b = (100.0, 100.0, 100.0, 100.0)
        self.assertEqual(measure.union(a, b), 10000.0)

        # Test two disjoint boxes.
        b = (100.0, 100.0, 200.0, 200.0)
        self.assertEqual(measure.union(a, b), 20000.0)

        # Test two partial joint union
        # Area of box a [100*100 = 10000], area of box b [100*100 = 10000]
        # Union = area box a + area box b - intersection(a, b) = 20000 - 2500 = 17500
        b = (50.0, 50.0, 150.0, 150.0)
        self.assertEqual(measure.union(a, b), 17500.0)


class TestHarmonicMean(unittest.TestCase):
    """Test weighted_harmonic_mean calculation."""

    def test_simple(self) -> None:
        """Test agreement between x, differing weights have no effect."""
        x = [1.0, 1.0]
        for i in range(1, 20):
            w = [i, 1.0]
            hm = measure.weighted_harmonic_mean(x, w)
            self.assertEqual(hm, 1.0)

    def test_different_x(self) -> None:
        """Test different x's, now weights are important."""
        x = [2.0, 1.0]

        w = [2.0, 1.0]
        hm1 = measure.weighted_harmonic_mean(x, w)

        w = [1.0, 2.0]
        hm2 = measure.weighted_harmonic_mean(x, w)

        self.assertEqual(hm1 > hm2, True)  # hm1 should be larger b/c larger weight is given to the larger value of x.

    def test_math(self) -> None:
        """Test the math calculation."""
        x = [1.5, 1.25]
        w = [2.0, 1.0]
        hm = measure.weighted_harmonic_mean(x, w)
        self.assertEqual(hm, 1.40625)

    def test_cornercases(self) -> None:
        """Test corner case of zeros."""
        x = [0.0, 0.0]
        w = [2.0, 1.0]
        hm = measure.weighted_harmonic_mean(x, w)
        self.assertEqual(hm, 0)


class TestLongLatDecomp(unittest.TestCase):
    """Test long_lat_dist_decomposition."""

    def test_euclidean(self) -> None:
        """
        Test if distance between gt and est is correctly decomposed as longitudinal and lateral components.
        This tests only checks the magnitude of both components.
        """
        for _ in range(5):
            gt: npt.NDArray[np.float64] = np.random.rand(2) * np.random.randint(10)
            est: npt.NDArray[np.float64] = np.random.rand(2) * np.random.randint(10)
            long, lat = measure.long_lat_dist_decomposition(gt, est)
            dist1 = np.linalg.norm([long, lat])
            dist2 = np.linalg.norm(gt - est)
            self.assertTrue(np.allclose(dist1, dist2))

    def test_trivial(self) -> None:
        """Test for two identical vectors."""
        gt = np.array([1, 1])  # type: ignore
        est = np.array([1, 1])  # type: ignore
        long_lat = measure.long_lat_dist_decomposition(gt, est)
        self.assertTrue(np.allclose(long_lat, (0, 0)))

    def test_zero(self) -> None:
        """Test for two zero vectors."""
        gt = np.array([0, 0])  # type: ignore
        est = np.array([0, 0])  # type: ignore
        long_lat = measure.long_lat_dist_decomposition(gt, est)
        self.assertTrue(np.allclose(long_lat, (0, 0)))

    def test_gt_x_axis(self) -> None:
        """Test for simple cases where gt_vector is on x axis."""
        # test 1
        gt = np.array([1, 0])  # type: ignore
        est = np.array([0, 1])  # type: ignore
        long_lat = measure.long_lat_dist_decomposition(gt, est)
        self.assertTrue(np.allclose(long_lat, (-1, 1)))
        # test 2
        gt = np.array([1, 0])
        est = np.array([1, 1])
        long_lat = measure.long_lat_dist_decomposition(gt, est)
        self.assertTrue(np.allclose(long_lat, (0, 1)))
        # test 3
        gt = np.array([1, 0])
        est = np.array([3, 4])
        long_lat = measure.long_lat_dist_decomposition(gt, est)
        self.assertTrue(np.allclose(long_lat, (2, 4)))

    def test_negative(self) -> None:
        """Test when both gt and est are in negative directions."""
        # test 1
        gt = np.array([-1, -1])  # type: ignore
        est = np.array([-2, -2])  # type: ignore
        long_lat = measure.long_lat_dist_decomposition(gt, est)
        self.assertTrue(np.allclose(long_lat, (np.sqrt(2), 0)))
        # test 2
        gt = np.array([-1, -1])
        est = np.array([-1, 0])
        long_lat = measure.long_lat_dist_decomposition(gt, est)
        self.assertTrue(long_lat, (1 / np.sqrt(2), 1 / np.sqrt(2)))

    def test_edge_case(self) -> None:
        """Test some edge cases."""
        # Same direction
        gt = np.array([1, 1])  # type: ignore
        est = np.array([2, 2])  # type: ignore
        long_lat = measure.long_lat_dist_decomposition(gt, est)
        self.assertTrue(np.allclose(long_lat, (np.sqrt(2), 0)))

        # Opposite direction
        gt = np.array([-1, -1])
        est = np.array([1, 1])
        long_lat = measure.long_lat_dist_decomposition(gt, est)
        self.assertTrue(np.allclose(long_lat, (-np.sqrt(8), 0)))


if __name__ == '__main__':
    unittest.main()
