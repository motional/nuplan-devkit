import gc
import pickle
import unittest

import guppy

from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.tests.test_utils_nuplan_db import get_test_nuplan_db_wrapper_nocache


class TestNuPlanDBWrapper(unittest.TestCase):
    """Test NuPlanDB wrapper which supports loading/accessing multiple log databases."""

    def setUp(self) -> None:
        """Set up test case."""
        # Using the cached version of this method can interfere with the memory test
        self.db_wrapper = get_test_nuplan_db_wrapper_nocache()

    def test_serialization(self) -> None:
        """Test whether the wrapper object can be serialized/deserialized correctly."""
        serialized_binary = pickle.dumps(self.db_wrapper)
        re_db_wrapper: NuPlanDBWrapper = pickle.loads(serialized_binary)

        self.assertEqual(self.db_wrapper.data_root, re_db_wrapper.data_root)

    def test_maps_db(self) -> None:
        """Test that maps DB has been loaded."""
        self.db_wrapper.maps_db.load_vector_layer('us-nv-las-vegas-strip', 'lane_connectors')

    def test_nuplandb_wrapper_memory_usage(self) -> None:
        """
        Test that repeatedly creating and destroying nuplan DB wrapper objects does not cause memory leaks.
        """

        def spin_up_db_wrapper() -> None:
            db_wrapper = get_test_nuplan_db_wrapper_nocache()

            # Not strictly necessary, but avoid linter errors
            del db_wrapper

        starting_usage = 0
        ending_usage = 0
        num_iterations = 5  # Use a few extra iterations to allow maps_db caches to fill

        hpy = guppy.hpy()
        hpy.setrelheap()

        for i in range(0, num_iterations, 1):
            # Use nested function to ensure local handles go out of scope
            spin_up_db_wrapper()
            gc.collect()

            heap = hpy.heap()

            # Force heapy to materialize the heap statistics
            # This is done lasily, which can lead to noise if not forced.
            _ = heap.size

            if i == num_iterations - 2:
                starting_usage = heap.size
            if i == num_iterations - 1:
                ending_usage = heap.size

        memory_difference_in_mb = (ending_usage - starting_usage) / (1024 * 1024)

        # Alert on either 100 kb growth or 10 % of starting usage, whichever is bigger
        max_allowable_growth_mb = max(0.1, 0.1 * starting_usage / (1024 * 1024))
        self.assertGreater(max_allowable_growth_mb, memory_difference_in_mb)


if __name__ == '__main__':
    unittest.main()
