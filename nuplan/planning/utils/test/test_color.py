import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from nuplan.planning.utils.color import Color, ColorType


class TestColor(TestCase):
    """
    Test color.
    """

    def setUp(self) -> None:
        """
        Set up.
        """
        self.red = 0.1
        self.green = 0.2
        self.blue = 0.3
        self.alpha = 0.5

        self.color = Color(self.red, self.green, self.blue, self.alpha, ColorType.FLOAT)
        self.color_255 = Color(self.red, self.green, self.blue, self.alpha, ColorType.INT)

    def test_init(self) -> None:
        """
        Test initialisation.
        """
        # Assertions
        self.assertEqual(self.color.red, self.red)
        self.assertEqual(self.color.green, self.green)
        self.assertEqual(self.color.blue, self.blue)
        self.assertEqual(self.color.alpha, self.alpha)

        self.assertEqual(self.color.serialize_to, ColorType.FLOAT)
        self.assertEqual(self.color_255.serialize_to, ColorType.INT)

    def test_post_init_invalid_type(self) -> None:
        """
        Tests that post init raises TypeError when passing any non-float types.
        """
        # Assertions
        with self.assertRaises(TypeError):
            Color(1.0, 0.5, 0.0, '1')

    def test_post_init_invalid_range(self) -> None:
        """
        Tests that post init raises ValueError when passing values outside of range 0-255.
        """
        # Assertions
        with self.assertRaises(ValueError):
            Color(1.0, 0.5, 0.0, 100.0)
        with self.assertRaises(ValueError):
            Color(1.0, 0.5, 0.0, -1.0)

    def test_iter(self) -> None:
        """
        Tests iteration of RGBA components.
        """
        # Call method under test
        result = [color for color in self.color]

        # Assertions
        self.assertEqual(result[0], self.red)
        self.assertEqual(result[1], self.green)
        self.assertEqual(result[2], self.blue)
        self.assertEqual(result[3], self.alpha)

    def test_iter_255(self) -> None:
        """
        Tests iteration of RGBA components, with color type specified as int.
        """
        # Call method under test
        result = [color for color in self.color_255]

        # Assertions
        self.assertEqual(result[0], int(self.red * 255))
        self.assertEqual(result[1], int(self.green * 255))
        self.assertEqual(result[2], int(self.blue * 255))
        self.assertEqual(result[3], int(self.alpha * 255))

    def test_to_list(self) -> None:
        """
        Tests to list method.
        """
        # Call method under test
        result = self.color.to_list()

        # Assertions
        self.assertEqual(result, [self.red, self.green, self.blue, self.alpha])

    def test_mul(self) -> None:
        """
        Tests multiplication operation without clamping ie. results already in range (0-255).
        """
        # Call method under test
        result = self.color * 2

        # Assertions
        self.assertEqual(result, Color(self.red * 2, self.green * 2, self.blue * 2, self.alpha * 2))

    def test_mul_clamp(self) -> None:
        """
        Tests clamping of values to range (0-255) after multiplication.
        """
        # Set up
        red = 0.5
        green = 0.7
        blue = 0.0
        alpha = 1.0

        # Call method under test
        color = Color(red, green, blue, alpha) * 2

        # Assertions
        self.assertEqual(color.red, 1.0)
        self.assertEqual(color.green, 1.0)
        self.assertEqual(color.blue, 0.0)
        self.assertEqual(color.alpha, 1.0)

    def test_mul_255(self) -> None:
        """
        Tests multiplication operation with a color of integer color type preserves color type
        """
        # Call method under test
        result = self.color_255 * 2

        # Assertions
        self.assertEqual(result, Color(self.red * 2, self.green * 2, self.blue * 2, self.alpha * 2, ColorType.INT))

    @patch("nuplan.planning.utils.color.Color.__mul__")
    def test_rmul(self, mock_mul: Mock) -> None:
        """
        Tests reverse multiplication operation.
        """
        # Call method under test
        result = 2 * self.color

        # Assertions
        mock_mul.assert_called_once_with(2)
        self.assertEqual(result, mock_mul.return_value)


if __name__ == '__main__':
    unittest.main()
