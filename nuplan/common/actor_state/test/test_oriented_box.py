import math as m
import unittest

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2


class TestOrientedBox(unittest.TestCase):
    """Tests OrientedBox class"""

    def setUp(self) -> None:
        """Creates sample parameters for testing"""
        self.center = StateSE2(1, 2, m.pi / 8)
        self.length = 4.0
        self.width = 2.0
        self.height = 1.5
        self.expected_vertices = [(2.47, 3.69), (-1.23, 2.16), (-0.47, 0.31), (3.23, 1.84)]

    def test_construction(self) -> None:
        """Tests that the object is created correctly, including the polygon representing its geometry."""
        test_box = OrientedBox(self.center, self.length, self.width, self.height)

        self.assertTrue(self.center == test_box.center)
        self.assertEqual(self.length, test_box.length)
        self.assertEqual(self.width, test_box.width)
        self.assertEqual(self.height, test_box.height)

        # Check lazy loading is working
        self.assertFalse("geometry" in test_box.__dict__)

        for vertex, expected_vertex in zip(test_box.geometry.exterior.coords, self.expected_vertices):
            self.assertAlmostEqual(vertex[0], expected_vertex[0], 2)
            self.assertAlmostEqual(vertex[1], expected_vertex[1], 2)

        # Check lazy loading is working
        self.assertTrue("geometry" in test_box.__dict__)


if __name__ == '__main__':
    unittest.main()
