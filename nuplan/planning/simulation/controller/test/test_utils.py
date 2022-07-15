import unittest

import numpy as np

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.controller.utils import forward_integrate


class TestUtils(unittest.TestCase):
    """
    Tests utils library.
    """

    def setUp(self) -> None:
        """Sets sample parameters for testing."""
        np.random.seed(0)
        self.inits = np.random.rand(100)
        self.deltas = np.random.rand(100)
        self.sampling_times = np.random.randint(1000000, size=100)

    def test_forward_integrate(self) -> None:
        """
        Test forward_integrate.
        """
        for init, delta, sampling_time in zip(self.inits, self.deltas, self.sampling_times):
            result = forward_integrate(init, delta, TimePoint(sampling_time))
            expect = init + delta * sampling_time * 1e-6
            self.assertAlmostEqual(result, expect)


if __name__ == '__main__':
    unittest.main()
