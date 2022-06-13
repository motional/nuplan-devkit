import unittest
import unittest.mock as mock

import numpy as np
from PIL import Image, ImageDraw

from nuplan.database.utils.plot import build_color_mask, draw_masks, image_with_boxes, rainbow


class TestRainbow(unittest.TestCase):
    """Test the rainbow."""

    def test_number_colors(self) -> None:
        """Check that correct number of colors is returned."""
        n_list = [3, 5, 7]
        for n in n_list:
            colors = rainbow(n)
            self.assertEqual(len(colors), n)

    def test_normalized(self) -> None:
        """Check that the colors are normalized."""
        n = 7
        colors = rainbow(n, normalized=True)

        # Confirm type and range of values.
        for color in colors:
            for c in color:
                self.assertTrue(isinstance(c, float))
                self.assertTrue(0.0 <= c <= 1.0)

    def test_non_normalized(self) -> None:
        """Check that the colors are not normalized."""
        n = 7
        colors = rainbow(n, normalized=False)

        # Confirm type and range of values.
        for color in colors:
            for c in color:
                self.assertTrue(isinstance(c, int))
                self.assertTrue(0 <= c <= 255)

        # Max sure values are large enough.
        max_value = max([max(color) for color in colors])
        self.assertTrue(max_value > 1)


class TestBuildColorMask(unittest.TestCase):
    """Test build color mask function."""

    def test_build_color_mask(self) -> None:
        """Check if correct color mask is built."""
        # color dictionary
        colors = {0: (0, 0, 0, 0), 1: (128, 20, 20, 10), 2: (255, 100, 100, 255)}

        # sample input
        test_array = np.array([[0, 1], [2, 2]])  # type: ignore

        # target mask for sample input
        target_mask = np.array(
            [[[0, 0, 0, 0], [128, 20, 20, 10]], [[255, 100, 100, 255], [255, 100, 100, 255]]]
        )  # type: ignore

        color_mask = build_color_mask(test_array, colors)
        self.assertEqual(np.array_equal(color_mask, target_mask), True)

    def test_build_color_mask_invalid_key(self) -> None:
        """Check if build_color_mask throws a KeyError exception for invalid keys."""
        # color dictionary
        colors = {100: (0, 0, 0, 0), 1: (128, 20, 20, 10), 2: (255, 100, 100, 255)}

        # sample input
        test_array = np.array([[0, 1], [2, 2]])  # type: ignore

        with self.assertRaises(KeyError):
            build_color_mask(test_array, colors)


class TestImageWithBoxes(unittest.TestCase):
    """Test drawing of input image with boxes and labels."""

    @mock.patch('nuplan.database.utils.plot._color_prep')
    def test_image_with_boxes(self, mock__color_prep) -> None:  # type: ignore
        """Test function of viewing image with boxes."""
        # build test image
        target_image_array = np.zeros((100, 100, 3), np.uint8)  # type: ignore
        target_image_array[10:41, 10:41] = (255, 0, 0)
        target_image_array[60:91, 60:91] = (0, 255, 0)

        # Remove top right and bottom left corner from rectangles as in tested function
        target_image_array[10, 40] = (0, 0, 0)
        target_image_array[40, 10] = (0, 0, 0)
        target_image_array[60, 90] = (0, 0, 0)
        target_image_array[90, 60] = (0, 0, 0)

        target_image = Image.fromarray(target_image_array)
        draw = ImageDraw.Draw(target_image, 'RGB')

        test_image = np.zeros((100, 100, 3), np.uint8)  # type: ignore

        # dummy labelset
        labelset = {3: "green", 2: "red"}

        # list of detected boxes
        boxes = [(11.0, 11.0, 40.0, 40.0), (61.0, 61.0, 90.0, 90.0)]

        # list of scores
        scores = [0.01, 0.01]

        # list of labels
        labels = [2, 3]

        # dummy color dictionary
        colors = {2: (255, 0, 0, 255), 3: (0, 255, 0, 255)}
        # mock color dictionary creation
        mock__color_prep.return_value = colors

        # draw text on target boxes
        draw.text((boxes[0][0], boxes[0][1]), 'red: 1')
        draw.text((boxes[1][0], boxes[1][1]), 'green: 1')

        image = image_with_boxes(test_image, boxes, labels, 2, 255, labelset, scores, colors)
        image = np.array(image.convert('RGB'))
        target_image_converted = np.array(target_image.convert('RGB'))  # type: ignore

        mock__color_prep.assert_called_with(2, 255, colors)
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(target_image_converted, np.ndarray)
        self.assertEqual(np.array_equal(target_image_converted, image), True)


class TestDrawMasks(unittest.TestCase):
    """Test draw_masks function."""

    @mock.patch("nuplan.database.utils.plot._color_prep")
    def test_draw_masks(self, mock__color_prep) -> None:  # type: ignore
        """Test Drawing Masks on Image."""
        # create test image with different intensity squares
        # Top left square -> Red
        # Top right square -> Green
        # Bottom left square -> Blue
        # Bottom right square -> White
        test_image = np.zeros((100, 100, 3), np.uint8)  # type: ignore
        test_image[0:50, 0:50, :] = (255, 0, 0)
        test_image[0:50, 50:100, :] = (0, 255, 0)
        test_image[50:100, 0:50, :] = (0, 0, 255)
        test_image[50:, 50:, :] = (255, 255, 255)

        # assign different class label to each square
        test_target = np.zeros((100, 100))
        test_target[:50, :50] = 1
        test_target[:50, 50:] = 2
        test_target[50:, :50] = 3
        test_target[50:, 50:] = 4

        # dummy color dictionary
        colors = {1: (255, 0, 0, 255), 2: (0, 255, 0, 255), 3: (0, 0, 255, 255), 4: (255, 255, 255, 255)}
        # mock color dictionary creation
        mock__color_prep.return_value = colors

        # Input image for drawing mask is all zeros
        input_image = Image.fromarray(np.zeros((100, 100, 3), np.uint8))

        # masks should be drawn with full opacity
        draw_masks(input_image, test_target, ncolors=4, colors=colors, alpha=255)
        mock__color_prep.assert_called_with(4, 255, colors)
        input_image = np.array(input_image)
        self.assertEqual(np.array_equal(input_image, test_image), True)


if __name__ == '__main__':
    unittest.main()
