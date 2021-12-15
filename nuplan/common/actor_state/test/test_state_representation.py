import unittest

from nuplan.common.actor_state.state_representation import Point2D, StateSE2, TimePoint


class TestStateRepresentation(unittest.TestCase):
    """ Test StateSE2 and Point2D """

    def test_time_point(self) -> None:
        t1 = TimePoint(123123)
        t2 = TimePoint(234234)

        with self.assertRaises(AssertionError):
            _ = TimePoint(-42)
        self.assertTrue(t2 > t1)
        self.assertTrue(t1 < t2)
        self.assertEqual(0.123123, t1.time_s)
        self.assertEqual(TimePoint(357357), t1 + t2)
        self.assertEqual(TimePoint(111111), t2 - t1)

    def test_point2d(self) -> None:
        """ Test Point2D """
        x = 1.2222
        y = 3.553435
        point = Point2D(x=x, y=y)
        self.assertAlmostEqual(point.x, x)
        self.assertAlmostEqual(point.y, y)

    def test_state_se2(self) -> None:
        """ Test StateSE2 """

        x = 1.2222
        y = 3.553435
        heading = 1.32498
        state = StateSE2(x, y, heading)

        self.assertAlmostEqual(state.x, x)
        self.assertAlmostEqual(state.y, y)
        self.assertAlmostEqual(state.heading, heading)


if __name__ == '__main__':
    unittest.main()
