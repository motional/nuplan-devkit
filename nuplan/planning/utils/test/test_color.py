import unittest
from unittest import TestCase

from nuplan.planning.utils.color import Color


class TestColor(TestCase):
    """
    Test color
    """

    def setUp(self) -> None:
        """Setup the test."""
        self.red = 1
        self.green = 2
        self.blue = 3
        self.alpha = 4

        self.color = Color(self.red, self.green, self.blue, self.alpha)

    def test_init(self) -> None:
        """
        Actually testing both init and all the getters
        of the Color class, as it's a dataclass.
        Assertions
        """
        self.assertEqual(self.color.red, self.red)
        self.assertEqual(self.color.green, self.green)
        self.assertEqual(self.color.blue, self.blue)
        self.assertEqual(self.color.alpha, self.alpha)

    def test_to_list(self) -> None:
        """Call method under test."""
        result = self.color.to_list()

        # Assertions
        self.assertEqual(result, [self.red, self.green, self.blue, self.alpha])


if __name__ == '__main__':
    unittest.main()
