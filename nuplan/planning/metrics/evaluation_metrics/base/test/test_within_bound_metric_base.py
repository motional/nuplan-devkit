import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt

from nuplan.planning.metrics.evaluation_metrics.base.within_bound_metric_base import WithinBoundMetricBase


class TestWithinBoundMetricBase(TestCase):
    """
    Test WithinBoundMetricBase
    """

    def setUp(self) -> None:
        """
        Set up the test
        """
        values: npt.NDArray[np.float32] = np.array(np.random.normal(size=(10,)))
        self.max_val = np.max(values) + 1e-4
        self.min_val = np.min(values) - 1e-4
        self.time_series = MagicMock()
        self.time_series.values = values.tolist()
        self.metrics = WithinBoundMetricBase(name='test', category='test')

    def test_compute_within_bound(self) -> None:
        """
        Test within bound metric
        """
        self.assertTrue(
            self.metrics._compute_within_bound(
                self.time_series, min_within_bound_threshold=self.min_val, max_within_bound_threshold=self.max_val
            )
        )

        self.assertFalse(
            self.metrics._compute_within_bound(
                self.time_series, min_within_bound_threshold=self.min_val + 0.1, max_within_bound_threshold=self.max_val
            )
        )

        self.assertFalse(
            self.metrics._compute_within_bound(
                self.time_series, min_within_bound_threshold=self.min_val, max_within_bound_threshold=self.max_val - 0.1
            )
        )

        self.assertFalse(
            self.metrics._compute_within_bound(
                self.time_series,
                min_within_bound_threshold=self.min_val + 0.1,
                max_within_bound_threshold=self.max_val - 0.1,
            )
        )


if __name__ == '__main__':
    unittest.main()
