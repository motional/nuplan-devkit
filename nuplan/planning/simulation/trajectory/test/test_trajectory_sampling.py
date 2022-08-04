import unittest

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class TestTrajectorySampling(unittest.TestCase):
    """
    Test trajectory sampling parameters
    """

    def test_wrong_setup(self) -> None:
        """Raise in case the sampling args are not consistent."""
        with self.assertRaises(ValueError):
            TrajectorySampling(num_poses=10, time_horizon=8, interval_length=0.5)

        with self.assertRaises(ValueError):
            TrajectorySampling()

        with self.assertRaises(ValueError):
            TrajectorySampling(num_poses=10)
            TrajectorySampling(time_horizon=10)
            TrajectorySampling(interval_length=10)

    def test_num_poses(self) -> None:
        """Test that num poses are set correctly."""
        sampling = TrajectorySampling(time_horizon=8, interval_length=0.5)
        self.assertEqual(sampling.time_horizon, 8)
        self.assertEqual(sampling.interval_length, 0.5)
        self.assertEqual(sampling.num_poses, 16)

    def test_num_poses_floating(self) -> None:
        """Test that num poses are set correctly even with floating point numbers."""
        sampling = TrajectorySampling(time_horizon=0.5, interval_length=0.1)
        self.assertEqual(sampling.time_horizon, 0.5)
        self.assertEqual(sampling.interval_length, 0.1)
        self.assertEqual(sampling.num_poses, 5)

    def test_interval(self) -> None:
        """Test that interval are set correctly."""
        sampling = TrajectorySampling(time_horizon=8, num_poses=16)
        self.assertEqual(sampling.time_horizon, 8)
        self.assertEqual(sampling.interval_length, 0.5)
        self.assertEqual(sampling.num_poses, 16)

    def test_time_horizon(self) -> None:
        """Test that time_horizon are set correctly."""
        sampling = TrajectorySampling(interval_length=0.5, num_poses=16)
        self.assertEqual(sampling.time_horizon, 8)
        self.assertEqual(sampling.interval_length, 0.5)
        self.assertEqual(sampling.num_poses, 16)


if __name__ == '__main__':
    unittest.main()
