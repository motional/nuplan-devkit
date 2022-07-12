import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from nuplan.planning.utils.color import Color


class TestColor(TestCase):
    """
    Test color.
    """

    def setUp(self) -> None:
        """
        Set up.
        """
        self.red = 1
        self.green = 2
        self.blue = 3
        self.alpha = 4

        self.color = Color(self.red, self.green, self.blue, self.alpha)

    def test_init(self) -> None:
        """
        Test initialisation.
        """
        # Assertions
        self.assertEqual(self.color.red, self.red)
        self.assertEqual(self.color.green, self.green)
        self.assertEqual(self.color.blue, self.blue)
        self.assertEqual(self.color.alpha, self.alpha)

    def test_post_init_invalid_type(self) -> None:
        """
        Tests that post init raises TypeError when passing any non-integer types.
        """
        # Assertions
        with self.assertRaises(TypeError):
            Color(1, 2, 3, 4.0)

    def test_post_init_invalid_range(self) -> None:
        """
        Tests that post init raises ValueError when passing values outside of range 0-255.
        """
        # Assertions
        with self.assertRaises(ValueError):
            Color(1, 2, 3, 400)
        with self.assertRaises(ValueError):
            Color(-1, 2, 3, 4)

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
        result = self.color * 10

        # Assertions
        self.assertEqual(result, Color(self.red * 10, self.green * 10, self.blue * 10, self.alpha * 10))

    def test_mul_clamp(self) -> None:
        """
        Tests clamping of values to range (0-255) after multiplication.
        """
        # Set up
        red = 100
        green = 200
        blue = 10
        alpha = 0

        # Call method under test
        color = Color(red, green, blue, alpha) * 2

        # Assertions
        self.assertEqual(color.red, 200)
        self.assertEqual(color.green, 255)
        self.assertEqual(color.blue, 20)
        self.assertEqual(color.alpha, 0)

    @patch("nuplan.planning.utils.color.Color.__mul__")
    def test_rmul(self, mock_mul: Mock) -> None:
        """
        Tests reverse multiplication operation.
        """
        # Call method under test
        result = 255 * self.color

        # Assertions
        mock_mul.assert_called_once_with(255)
        self.assertEqual(result, mock_mul.return_value)


if __name__ == '__main__':
    unittest.main()
