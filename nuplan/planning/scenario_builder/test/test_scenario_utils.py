import unittest

import numpy as np

from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon


class TestIndexTimeSampling(unittest.TestCase):
    """
    Tests the index time sampling functionality.
    """

    def test_round_time_horizon(self) -> None:
        """
        Tests the conversion of N number of samples and T time horizon (round) to sample indices.
        """
        time_interval = 0.05
        frames = np.arange(0, 20, time_interval)
        indices = sample_indices_with_time_horizon(num_samples=10, time_horizon=8.0, time_interval=time_interval)
        samples = frames[indices]

        assert np.allclose(samples, np.array([0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0]))

    def test_non_round_time_horizon(self) -> None:
        """
        Tests the conversion of N number of samples and T time horizon (non-round) to sample indices.
        """
        time_interval = 0.05
        frames = np.arange(0, 20, time_interval)
        indices = sample_indices_with_time_horizon(num_samples=12, time_horizon=1.2, time_interval=time_interval)
        samples = frames[indices]

        assert np.allclose(samples, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]))

    def test_raises_error(self) -> None:
        """
        Tests the edge case of receiving a smaller time horizon than time interval.
        """
        self.assertRaises(
            ValueError, sample_indices_with_time_horizon, num_samples=3, time_horizon=0.3, time_interval=0.5
        )


if __name__ == '__main__':
    unittest.main()
