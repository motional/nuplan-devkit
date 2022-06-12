import unittest
from typing import Any

import numpy as np
import numpy.typing as npt

from nuplan.database.maps_db.layer import MapLayer
from nuplan.database.maps_db.metadata import MapLayerMeta
from nuplan.database.maps_db.utils import compute_joint_distance_matrix

ONE_PIXEL_PER_METER = 1
FIVE_PIXELS_PER_METER = 0.2
TEN_PIXELS_PER_METER = 0.1

# Use 101x101 so the valid pixel values are 0 to 100, inclusive.
DUMMY_DATA_101_BY_101 = np.zeros((101, 101))


def make_dilatable_map_layer(mask: npt.NDArray[np.uint8], precision: float) -> MapLayer:
    """
    Convenience method for constructing a dilatable map with the appropriate pre-computed distances.
    :param mask: Pixel values.
    :param precision: Meters per pixel.
    :return: A MapLayer Object.
    """
    joint_distance = compute_joint_distance_matrix(mask, precision)
    layer = MapLayer(mask, make_meta(True, precision), joint_distance=joint_distance)

    return layer


def make_meta(can_dilate: bool, precision: float, is_binary: bool = True) -> MapLayerMeta:
    """
    Helper method to initialize a MapLayerMeta instance.
    :param can_dilate: whether to can dilate or not.
    :param precision: Meters per pixel.
    :param is_binary: Flag to indicate if is binary.
    :return: A MapLayerMeta object.
    """
    return MapLayerMeta(
        name='test_fixture', md5_hash='not used here', can_dilate=can_dilate, is_binary=is_binary, precision=precision
    )


class TestMask(unittest.TestCase):
    """Test Mask."""

    data = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])  # type: ignore

    def test_negative_dilation(self) -> None:
        """Checks if it raises with negative dilation values."""
        layer = make_dilatable_map_layer(TestMask.data, ONE_PIXEL_PER_METER)
        for negative_number in [-0.01, -1, -200]:
            with self.subTest(negative_number=negative_number):
                self.assertRaises(AssertionError, layer.mask, dilation=negative_number)

    def test_dilate_undilatable_layer(self) -> None:
        """Checks if it raises with dilating on unlidatable layer."""
        meta = make_meta(can_dilate=False, precision=ONE_PIXEL_PER_METER)
        layer = MapLayer(TestMask.data, meta)
        self.assertRaises(AssertionError, layer.mask, dilation=1)

    def test_no_dilation(self) -> None:
        """Tests layer with no dilation."""
        layer = make_dilatable_map_layer(TestMask.data, ONE_PIXEL_PER_METER)
        self.assertTrue(np.array_equal(layer.mask(dilation=0), TestMask.data))

    def test_dilation(self) -> None:
        """Tests dilation with different dilation values."""
        layer = make_dilatable_map_layer(TestMask.data, ONE_PIXEL_PER_METER)

        test_cases = [
            (0.1, np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])),
            (0.5, np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])),
            (1, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]])),
            (2, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
        ]  # type: ignore

        for dilation, expected_mask in test_cases:
            with self.subTest(dilation=dilation, expected_mask=expected_mask):
                self.assertTrue(np.array_equal(layer.mask(dilation=dilation), expected_mask))


class TestCrop(unittest.TestCase):
    """Test class for Cropping layer."""

    data = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])  # type: ignore

    def test_empty_slice(self) -> None:
        """Checks empty slice, and size of layer after cropping should be zero."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)
        crop = layer.crop(slice(0), slice(0))
        self.assertEqual(crop.shape, (0, 0))

    def test_full_slices(self) -> None:
        """Tests with full slice and various out-of-bounds slices."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)

        test_cases = [
            # full slice
            (slice(0, 3), slice(0, 3)),
            # various out-of-bounds slices
            (slice(0, 3), slice(0, 5)),
            (slice(0, 5), slice(0, 3)),
            (slice(0, 5), slice(0, 5)),
        ]
        for row_slice, col_slice in test_cases:
            with self.subTest(row_slice=row_slice, col_slice=col_slice):
                crop = layer.crop(row_slice, col_slice)
                self.assertTrue(np.array_equal(crop, TestCrop.data))

    def test_negative_dilation(self) -> None:
        """Tests to dilate with negative dilation value."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)
        for negative_number in [-0.01, -1, -200]:
            with self.subTest(negative_number=negative_number):
                self.assertRaises(
                    AssertionError, layer.crop, rows=slice(0, 2), cols=slice(0, 2), dilation=negative_number
                )

    def test_dilate_undilatable_layer(self) -> None:
        """Tests to dilate an undilatable layer in crop function."""
        meta = make_meta(can_dilate=False, precision=ONE_PIXEL_PER_METER)
        layer = MapLayer(TestCrop.data, meta)
        self.assertRaises(AssertionError, layer.crop, rows=slice(0, 2), cols=slice(0, 2), dilation=1)

    def test_no_dilation(self) -> None:
        """Test no dilation with crop function."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)
        upper_left_crop = layer.crop(rows=slice(0, 2), cols=slice(0, 2), dilation=0)
        self.assertTrue(np.array_equal(upper_left_crop, np.array([[1, 1], [1, 0]])))

        lower_right_crop = layer.crop(rows=slice(1, 3), cols=slice(1, 3), dilation=0)
        self.assertTrue(np.array_equal(lower_right_crop, np.array([[0, 0], [0, 0]])))

    def test_dilation(self) -> None:
        """Tests dilation in crop function."""
        layer = make_dilatable_map_layer(TestCrop.data, ONE_PIXEL_PER_METER)

        test_cases = [
            (0.1, np.array([[1, 0, 0], [0, 0, 0]])),
            (0.5, np.array([[1, 1, 0], [1, 0, 0]])),
            (1, np.array([[1, 1, 1], [1, 1, 0]])),
            (2, np.array([[1, 1, 1], [1, 1, 1]])),
        ]  # type: ignore

        for dilation, expected_lower_crop in test_cases:
            with self.subTest(dilation=dilation, expected_lower_crop=expected_lower_crop):
                crop = layer.crop(rows=slice(1, 3), cols=slice(0, 3), dilation=dilation)
                self.assertTrue(np.array_equal(crop, expected_lower_crop))


class TestTransformMatrix(unittest.TestCase):
    """Test Class for Transform matrix."""

    def test_transform_matrix(self) -> None:
        """Tests transform matrix for MapLayers with different precisions and different size."""
        test_cases = [
            (101, 101, 1, np.array([[1, 0, 0, 0], [0, -1, 0, 100], [0, 0, 1, 0], [0, 0, 0, 1]])),
            (101, 101, 0.1, np.array([[10, 0, 0, 0], [0, -10, 0, 100], [0, 0, 1, 0], [0, 0, 0, 1]])),
            (51, 51, 1, np.array([[1, 0, 0, 0], [0, -1, 0, 50], [0, 0, 1, 0], [0, 0, 0, 1]])),
            (51, 51, 10, np.array([[0.1, 0, 0, 0], [0, -0.1, 0, 50], [0, 0, 1, 0], [0, 0, 0, 1]])),
        ]  # type: ignore

        for nrows, ncols, precision, expected_matrix in test_cases:
            with self.subTest(nrows=nrows, ncols=ncols, precision=precision, expected_matrix=expected_matrix):
                layer = MapLayer(np.ones((nrows, ncols)), make_meta(False, precision))
                self.assertTrue(np.array_equal(layer.transform_matrix, expected_matrix))


class TestToPixelCoords(unittest.TestCase):
    """Test Class of converting to pixel coordinates."""

    def test_without_precision_scale(self) -> None:
        """Tests to_pixel_coords function on a map layer of 1 pixel per meter."""
        layer = MapLayer(DUMMY_DATA_101_BY_101, make_meta(False, ONE_PIXEL_PER_METER))

        # Each entry is an input point in space and its output point in pixels.
        test_cases = [
            # corners
            [(0, 0), (0, 100)],
            [(0, 100), (0, 0)],
            [(100, 0), (100, 100)],
            [(100, 100), (100, 0)],
            # edges
            [(0, 40), (0, 60)],
            [(40, 0), (40, 100)],
            [(40, 100), (40, 0)],
            [(100, 40), (100, 60)],
            # within bounds
            [(50, 50), (50, 50)],
            [(36, 42), (36, 58)],
            [(99, 37), (99, 63)],
            [(7, 99), (7, 1)],
        ]
        for input_point, expected_output in test_cases:
            with self.subTest(input_point=input_point, expected_output=expected_output):
                self.assertEqual(layer.to_pixel_coords(*input_point), expected_output)

    def test_with_precision_scale(self) -> None:
        """Test to_pixel_coords function on a map layer of 0.1 precision."""
        layer = MapLayer(DUMMY_DATA_101_BY_101, make_meta(False, TEN_PIXELS_PER_METER))

        # Each entry is an input point in space and its output point in pixels.
        test_cases = [
            # corners
            [(0, 0), (0, 100)],
            [(0, 10), (0, 0)],
            [(10, 0), (100, 100)],
            [(10, 10), (100, 0)],
            # edges
            [(0, 4), (0, 60)],
            [(4, 0), (40, 100)],
            [(4, 10), (40, 0)],
            [(10, 4), (100, 60)],
            # within bounds
            [(5, 5), (50, 50)],
            [(3.6, 4.2), (36, 58)],
            [(9.9, 3.7), (99, 63)],
            [(0.7, 9.9), (7, 1)],
        ]
        for input_point, expected_output in test_cases:  # type: ignore
            with self.subTest(input_point=input_point, expected_output=expected_output):  # type: ignore
                self.assertEqual(layer.to_pixel_coords(*input_point), expected_output)  # type: ignore

    def test_multiple_inputs(self) -> None:
        """Test with multiple inputs."""
        layer = MapLayer(DUMMY_DATA_101_BY_101, make_meta(False, ONE_PIXEL_PER_METER))

        input_x = np.array([5, 60])  # type: ignore
        input_y = np.array([20, 77])  # type: ignore
        expected_x = np.array([5, 60])  # type: ignore
        expected_y = np.array([80, 23])  # type: ignore

        output_x, output_y = layer.to_pixel_coords(input_x, input_y)
        self.assertTrue(np.array_equal(output_x, expected_x))
        self.assertTrue(np.array_equal(output_y, expected_y))


class OutOfBoundsData(object):
    """Class to define in-bounds and out-of-bounds data."""

    # Values between -0.0499 and .9499 are OK since they map to between [0, 9]
    # for a 10-by-10 layer using TEN_PIXELS_PER_METER.
    min_ = -0.0499
    max_ = 0.9499
    # Epsilon such that `min_ - e` and `max_ + e` are out of bounds.
    e = 0.0002
    in_bounds = np.array(
        [
            # Corners
            (min_, min_),
            (min_, max_),
            (max_, min_),
            (max_, max_),
            # On edges
            (min_, 0.44),
            (0.57, min_),
            (max_, 0.5),
            (0.94, max_),
            # Points in middle
            (0.123, 0.456),
            (0.62, 0.59),
            (0.1, 0.5),
        ]
    )  # type: ignore
    in_bounds_for_10_by_10_layer_with_10px_per_m = (in_bounds[:, 0], in_bounds[:, 1])

    out_of_bounds = np.array(
        [
            # Barely out of bounds
            (min_ - e, 0.44),
            (0.57, min_ - e),
            (max_ + e, 0.5),
            (0.94, max_ + e),
            # Far out of bounds
            (-1, 0.5),
            (0.2, -2),
            (10, 0.5),
            (0.8, 42),
            (14, 59),
        ]
    )  # type: ignore
    out_of_bounds_for_10_by_10_layer_with_10px_per_m = (out_of_bounds[:, 0], out_of_bounds[:, 1])


class TestIsOnMask(unittest.TestCase):
    """Test Class of is_on_mask function."""

    small_number = 0.00001  # Just a small number to avoid edge effects.
    half_gt = TEN_PIXELS_PER_METER / 2 + small_number  # Just larger than half a cell.
    half_lt = TEN_PIXELS_PER_METER / 2 - small_number  # Just smaller than half a cell.

    def test_out_of_bounds_without_dilation(self) -> None:
        """This checks the boundary conditions for is_on_mask."""
        mask = np.ones((10, 10))
        layer = MapLayer(mask, make_meta(False, TEN_PIXELS_PER_METER))

        x, y = OutOfBoundsData.in_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertTrue(np.all(layer.is_on_mask(x, y)))

        x, y = OutOfBoundsData.out_of_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertFalse(np.any(layer.is_on_mask(x, y)))

    def test_out_of_bounds_with_dilation(self) -> None:
        """This checks the boundary conditions for is_on_mask."""
        mask = np.ones((10, 10))
        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)  # type: ignore

        x, y = OutOfBoundsData.in_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertTrue(np.all(layer.is_on_mask(x, y, dilation=0.3)))

        x, y = OutOfBoundsData.out_of_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertFalse(np.any(layer.is_on_mask(x, y, dilation=0.3)))

    def test_native_resolution(self) -> None:
        """Test map resolution."""
        # Build a test map. 5 x 4 meters. All background except one pixel.
        mask = np.zeros((51, 40))

        # Native resolution is 0.1
        # Transformation in y is defined as y_pixel = (nrows - 1) - y_meters /  resolution
        # Transformation in x is defined as x_pixel = x_meters /  resolution
        # The global map location x=2, y=2 becomes row 30, column 20 in image map coords.
        mask[30, 20] = 1

        layer = MapLayer(mask, make_meta(False, TEN_PIXELS_PER_METER))

        # This is where we put the foreground in the fixture, so this should be true by design.
        self.assertTrue(layer.is_on_mask(2, 2))

        # Each pixel is 10 x 10 cm, so if we step less than 5 cm in either direction we are still on foreground.
        # Note that we add / subtract a "small number" to break numerical ambiguities along the edges.
        on_mask = [(2 + self.half_lt, 2), (2 - self.half_lt, 2), (2, 2 + self.half_lt), (2, 2 - self.half_lt)]

        # But if we step outside this range, we should get false
        off_mask = [(2 + self.half_gt, 2), (2 - self.half_gt, 2), (2, 2 + self.half_gt), (2, 2 - self.half_gt)]

        for point in on_mask:
            with self.subTest(point=point):
                self.assertTrue(layer.is_on_mask(*point)[0])

        for point in off_mask:
            with self.subTest(point=point):
                self.assertFalse(layer.is_on_mask(*point)[0])

    def test_edges(self) -> None:
        """Test map edges."""
        mask = np.ones((51, 40))

        layer = MapLayer(mask, make_meta(False, TEN_PIXELS_PER_METER))

        # Assert that corners are indeed drivable as encoded in map.
        self.assertTrue(layer.is_on_mask(0, 0.1))
        self.assertTrue(layer.is_on_mask(0, 5))
        self.assertTrue(layer.is_on_mask(3.9, 0.1))
        self.assertTrue(layer.is_on_mask(3.9, 5))

        # Now go just outside the map. This should no longer be drivable.
        self.assertFalse(layer.is_on_mask(3.9 + self.half_gt, 0.1))
        self.assertFalse(layer.is_on_mask(3.9 + self.half_gt, 5))
        self.assertFalse(layer.is_on_mask(0 - self.half_gt, 0.1))
        self.assertFalse(layer.is_on_mask(0 - self.half_gt, 5))


class TestDistToMask(unittest.TestCase):
    """This Class to test dist_to_mask function."""

    def test_out_of_bounds(self) -> None:
        """This checks the boundary conditions for dist_to_mask."""
        mask = np.ones((10, 10))
        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)  # type: ignore

        x, y = OutOfBoundsData.in_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertFalse(np.any(np.isnan(layer.dist_to_mask(x, y))))

        x, y = OutOfBoundsData.out_of_bounds_for_10_by_10_layer_with_10px_per_m
        self.assertTrue(np.all(np.isnan(layer.dist_to_mask(x, y))))

    def test_linear_edge_low_precision(self) -> None:
        """Tests linear edges with low precision of 1."""
        mask = np.array([[0, 0, 1, 1]])  # type: ignore
        layer = make_dilatable_map_layer(mask, ONE_PIXEL_PER_METER)

        test_cases = [
            (0, np.array([[1.5, 0.5, -0.5, -1.5]])),
            (1, np.array([[0.5, -0.5, -1.5, -2.5]])),
            (2, np.array([[-0.5, -1.5, -2.5, -3.5]])),
        ]  # type: ignore

        for dilation, expected_dist_to_mask in test_cases:
            for x in range(0, 4):
                with self.subTest(dilation=dilation, expected_dist_to_mask=expected_dist_to_mask, x=x):
                    actual = layer.dist_to_mask(x, 0, dilation)
                    expected = expected_dist_to_mask[0, x]
                    self.assertTrue(abs(actual - expected) < 0.001)

    def test_linear_edge_high_precision(self) -> None:
        """Test linear edge with high precision of 0.1."""
        mask = np.array([[0, 0, 1, 1]])  # type: ignore
        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)

        test_cases = [
            (0, np.array([[0.15, 0.05, -0.05, -0.15]])),
            (0.1, np.array([[0.05, -0.05, -0.15, -0.25]])),
            (0.2, np.array([[-0.05, -0.15, -0.25, -0.35]])),
        ]  # type: ignore

        x_in_meters = np.array([0, 0.1, 0.2, 0.3])  # type: ignore
        y_in_meters = np.array([0, 0, 0, 0])  # type: ignore

        for dilation, expected_dist_to_mask in test_cases:
            actual = layer.dist_to_mask(x_in_meters, y_in_meters, dilation)
            with self.subTest(dilation=dilation, expected_dist_to_mask=expected_dist_to_mask, actual=actual):
                self.assertTrue(np.allclose(actual, expected_dist_to_mask))

    def _test_non_linear_edge_helper(
        self, mask: npt.NDArray[np.uint8], expected_matrix: npt.NDArray[np.float64], test_name: str
    ) -> None:
        """
        Helper function to test nonlinear edge cases.
        :param mask: Pixel values.
        :param expected_matrix: The expected distance matrix of points on mask.
        :param test_name: A string of test name.
        """
        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)

        for dilation in [0, 0.1, 0.2, 0.3]:
            for x_in_pixels in range(0, 5):
                for y_in_pixels in range(0, 5):
                    x_in_meters = x_in_pixels * 0.1
                    y_in_meters = y_in_pixels * 0.1
                    matrix_row = mask.shape[1] - 1 - y_in_pixels
                    matrix_col = x_in_pixels
                    actual = layer.dist_to_mask(x_in_meters, y_in_meters, dilation=dilation)
                    expected = expected_matrix[matrix_row, matrix_col] - dilation
                    with self.subTest(
                        x_in_meters=x_in_meters,
                        y_in_meters=y_in_meters,
                        actual=actual,
                        expected=expected,
                        test_name=test_name,
                    ):
                        self.assertTrue(abs(actual - expected) < 0.005)

    def test_round_edge(self) -> None:
        """Test case of round edge."""
        mask = np.array(
            [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        )  # type: ignore
        # The distance between two cells *in pixels* is:
        # sqrt(change_in_x^2 + change_in_y^2) - 0.5
        # When a cell is 0, we give distance to closest 1.
        # When a cell is 1, we give distance to closest 0 as negative number.
        expected_matrix = np.array(
            [
                [0.266, 0.23, 0.15, 0.05, -0.05],
                [0.174, 0.15, 0.09, 0.05, -0.05],
                [0.09, 0.05, 0.05, -0.05, -0.09],
                [0.05, -0.05, -0.05, -0.09, -0.174],
                [-0.05, -0.09, -0.15, -0.174, -0.23],
            ]
        )  # type: ignore

        self._test_non_linear_edge_helper(mask, expected_matrix, 'test_round_edge')

    def test_hole(self) -> None:
        """Test case of a hole mask."""
        mask = np.array(
            [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
        )  # type: ignore
        expected_matrix = np.array(
            [
                [-0.09, -0.05, -0.05, -0.05, -0.09],
                [-0.05, 0.05, 0.05, 0.05, -0.05],
                [-0.05, 0.05, 0.15, 0.09, 0.05],
                [-0.05, 0.05, 0.05, 0.05, -0.05],
                [-0.09, -0.05, -0.05, -0.05, -0.09],
            ]
        )  # type: ignore

        self._test_non_linear_edge_helper(mask, expected_matrix, 'test_hole')


class TestConsistentIsOnMaskDistToMask(unittest.TestCase):
    """This Class to test the consistency of is_on_mask and dist_to_mask function."""

    def assert_on_mask_equals(self, layer: MapLayer, x: Any, y: Any, dilation: float, expected: bool) -> None:
        """
        Asserts when point is on mask, return True, otherwise False, and
        dist_to_mask gives a negative value, otherwise positive.
        :param layer: A MapLayer Object.
        :param x: Global x coordinates. Can be a scalar, list or a numpy array of x coordinates.
        :param y: Global y coordinates. Can be a scalar, list or a numpy array of x coordinates.
        :param dilation: Specifies the threshold on the distance from the drivable_area mask.
        The drivable_area mask is dilated to include points which are within this distance from itself.
        :param expected: The expected boolean value.
        """
        on_mask_result = layer.is_on_mask(x, y, dilation=dilation)
        self.assertEqual(
            on_mask_result,
            expected,
            f"expected is_on_mask({x}, {y}, {dilation}) to be {expected}, got {on_mask_result}",
        )
        # When point is on mask, dist_to_mask gives a negative value, otherwise positive.
        dist_to_mask_result = layer.dist_to_mask(x, y, dilation=dilation)
        self.assertEqual(
            dist_to_mask_result <= 0,
            expected,
            f"expected dist_to_mask({x}, {y}, {dilation}), to be "
            f"{'<= 0' if expected else '> 0'}, got {dist_to_mask_result}",
        )

    def test_dilation_with_foreground_point(self) -> None:
        """Test dilation with foreground point."""
        # Build a test map. 5 x 4 meters. All background except one pixel.
        mask = np.zeros((51, 40))

        # Native resolution is 0.1
        # Transformation in y is defined as y_pixel = (nrows - 1) - y_meters /  resolution
        # Transformation in x is defined as x_pixel = x_meters /  resolution
        # The global map location x=2, y=2 becomes row 30, column 20 in image map coords.
        mask[30, 20] = 1

        layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)  # type: ignore

        # This is where we put the foreground in the fixture, so this should be true by design.
        self.assertTrue(layer.is_on_mask(2, 2))

        # Go 1 meter to the right. Obviously not on the mask.
        self.assertFalse(layer.is_on_mask(2, 3))

        # But if we dilate by 1 meters, we are on the dilated mask.
        self.assert_on_mask_equals(layer, 2, 3, dilation=1, expected=True)  # y direction
        self.assert_on_mask_equals(layer, 3, 2, dilation=1, expected=True)  # x direction
        self.assert_on_mask_equals(layer, 2 + np.sqrt(1 / 2), 2 + np.sqrt(1 / 2), dilation=1, expected=True)  # diagonal
        # If we dilate by 0.9 meter, it is not enough.
        self.assert_on_mask_equals(layer, 2, 3, dilation=0.9, expected=False)

    def test_dilation_with_curved_line(self) -> None:
        """Test Dilation over Curved Line."""
        # This may be over-testing, but this test covers a slightly more
        # 'realistic' use case of giving a 'buffer' to an unsafe area.
        #
        # Consider this shape:
        #  000
        # 00000
        # 00X00
        # 00000
        #  000
        # All the 0's are at radius < 2.5 from the X. Therefore, in the data
        # below, only 2 leftmost entries in the top row are > 2.5 units from the
        # 1's.
        mask = np.array(
            [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        )  # type: ignore
        layer = make_dilatable_map_layer(mask, ONE_PIXEL_PER_METER)
        off_mask_points_in_physical_space = [(0, 4), (1, 4)]

        for x in range(0, 5):
            for y in range(0, 5):
                with self.subTest(x=x, y=y):
                    expected_on_mask = (x, y) not in off_mask_points_in_physical_space

                    # Dilation of 2m is effectively 2.5m when using
                    # ONE_PIXEL_PER_METER precision, since our code considers the
                    # mask to end half a pixel beyond the 1's.
                    self.assert_on_mask_equals(layer, x, y, dilation=2, expected=expected_on_mask)

    def test_coarse_resolution(self) -> None:
        """Tests with normal and low resolution."""
        # Due to resize that happens on load we need to inflate the fixture.
        mask = np.zeros((51, 40))
        mask[30, 20] = 1
        mask[31, 20] = 1
        mask[30, 21] = 1
        mask[31, 21] = 1

        normal_res_layer = make_dilatable_map_layer(mask, TEN_PIXELS_PER_METER)  # type: ignore
        low_res_layer = make_dilatable_map_layer(mask, FIVE_PIXELS_PER_METER)  # type: ignore

        # This is where we put the foreground in the fixture, so this should be true by design.
        self.assert_on_mask_equals(normal_res_layer, 2, 2, dilation=0, expected=True)

        # Go two meters to the right. Obviously not on the mask.
        self.assert_on_mask_equals(low_res_layer, 2, 4, dilation=0, expected=False)

        # But if we dilate by two meters, we are on the dilated mask.
        self.assert_on_mask_equals(low_res_layer, 2, 4, dilation=2.0, expected=True)

        # If we dilate by 1.9 meters, we are *still* the dilated mask, because
        # we made a design decision to have the mask extend half a pixel beyond
        # each 1.
        self.assert_on_mask_equals(low_res_layer, 2, 4, dilation=1.9001, expected=True)

        # If we dilate by 1.8 meters (9 pixels), the point is off the mask,
        # because our code considers it 1.9 meters (9.5 pixels) away from the
        # mask.
        self.assert_on_mask_equals(low_res_layer, 2, 4, dilation=1.8, expected=False)


if __name__ == '__main__':
    unittest.main()
