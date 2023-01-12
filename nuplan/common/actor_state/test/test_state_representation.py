import unittest

from nuplan.common.actor_state.state_representation import Point2D, StateSE2, TimeDuration, TimePoint


class TestTimeDuration(unittest.TestCase):
    """Tests for TimeDurationClass"""

    def test_default_initialization(self) -> None:
        """Checks raising when constructor is called directly unless flagged."""
        with self.assertRaises(RuntimeError):
            # It's not allowed to use the default constructor unless specially flagged
            _ = TimeDuration(time_us=42)
        dt = TimeDuration(time_us=42, _direct=False)
        self.assertEqual(dt.time_us, 42)

    def test_constructors(self) -> None:
        """Checks constructors perform correct conversions"""
        dt_s = TimeDuration.from_s(42)
        dt_ms = TimeDuration.from_ms(42)
        dt_us = TimeDuration.from_us(42)

        self.assertEqual(dt_s.time_us, 42000000)
        self.assertEqual(dt_ms.time_us, 42000)
        self.assertEqual(dt_us.time_us, 42)

    def test_getters(self) -> None:
        """Checks getters work as intended"""
        dt = TimeDuration.from_s(42)
        value_s = dt.time_s
        value_ms = dt.time_ms
        value_us = dt.time_us

        self.assertEqual(value_s, 42)
        self.assertEqual(value_ms, 42000)
        self.assertEqual(value_us, 42000000)

    def test_operators(self) -> None:
        """Tests basic math operators."""
        t1 = TimeDuration.from_s(1)
        t2 = TimeDuration.from_s(2)

        self.assertTrue(t2 > t1)
        self.assertFalse(t2 < t1)
        self.assertTrue(t1 < t2)
        self.assertFalse(t1 > t2)
        self.assertTrue(t1 == t1)
        self.assertFalse(t1 == t2)
        self.assertTrue(t1 >= t1)
        self.assertTrue(t1 <= t1)

        self.assertEqual((t1 + t2).time_s, 3)
        self.assertEqual((t1 - t2).time_s, -1)

        self.assertEqual((t1 * 3).time_s, 3)
        self.assertEqual((3 * t1).time_s, 3)

        self.assertEqual((t2 / 2).time_s, 1)
        self.assertEqual((t2 // 3).time_s, 0)


class TestTimePoint(unittest.TestCase):
    """Tests for TimePoint class."""

    def test_initialization(self) -> None:
        """Tests initialization fails with negative values and works otherwise."""
        with self.assertRaises(AssertionError):
            _ = TimePoint(-42)

        t1 = TimePoint(123456)
        self.assertEqual(t1.time_us, 123456)

    def test_comparisons(self) -> None:
        """Test basic comparison operators."""
        t1 = TimePoint(123123)
        t2 = TimePoint(234234)

        self.assertTrue(t2 > t1)
        self.assertFalse(t2 < t1)
        self.assertTrue(t1 < t2)
        self.assertFalse(t1 > t2)
        self.assertTrue(t1 == t1)
        self.assertFalse(t1 == t2)
        self.assertTrue(t1 >= t1)
        self.assertTrue(t1 <= t1)

    def test_addition(self) -> None:
        """Tests addition and subtractions."""
        t1 = TimePoint(123)
        dt = TimeDuration.from_us(100)

        self.assertEqual(t1 + dt, TimePoint(223))
        self.assertEqual(dt + t1, TimePoint(223))
        self.assertEqual(t1 - dt, TimePoint(23))


class TestStateRepresentation(unittest.TestCase):
    """Test StateSE2 and Point2D"""

    def test_point2d(self) -> None:
        """Test Point2D"""
        x = 1.2222
        y = 3.553435
        point = Point2D(x=x, y=y)
        self.assertAlmostEqual(point.x, x)
        self.assertAlmostEqual(point.y, y)

    def test_state_se2(self) -> None:
        """Test StateSE2"""
        x = 1.2222
        y = 3.553435
        heading = 1.32498
        state = StateSE2(x, y, heading)

        self.assertAlmostEqual(state.x, x)
        self.assertAlmostEqual(state.y, y)
        self.assertAlmostEqual(state.heading, heading)


if __name__ == '__main__':
    unittest.main()
