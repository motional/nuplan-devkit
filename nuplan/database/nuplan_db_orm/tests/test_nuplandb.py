import gc
import pickle
import unittest

import guppy

from nuplan.database.common.db import Table
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.tests.test_utils_nuplan_db import get_test_nuplan_db, get_test_nuplan_db_nocache


class TestNuPlanDB(unittest.TestCase):
    """Test main nuPlan database class."""

    def setUp(self) -> None:
        """Set up test case."""
        self.db = get_test_nuplan_db()
        self.db.add_ref()

    def test_pickle(self) -> None:
        """Test dumping and loading the object through pickle."""
        db_binary = pickle.dumps(self.db)
        re_db: NuPlanDB = pickle.loads(db_binary)

        self.assertEqual(self.db.data_root, re_db.data_root)
        self.assertEqual(self.db.name, re_db.name)
        self.assertEqual(self.db._verbose, re_db._verbose)

    def test_table_getters(self) -> None:
        """Test the table getters."""
        self.assertTrue(isinstance(self.db.category, Table))
        self.assertTrue(isinstance(self.db.camera, Table))
        self.assertTrue(isinstance(self.db.lidar, Table))
        self.assertTrue(isinstance(self.db.image, Table))
        self.assertTrue(isinstance(self.db.lidar_pc, Table))
        self.assertTrue(isinstance(self.db.lidar_box, Table))
        self.assertTrue(isinstance(self.db.track, Table))
        self.assertTrue(isinstance(self.db.scene, Table))
        self.assertTrue(isinstance(self.db.scenario_tag, Table))
        self.assertTrue(isinstance(self.db.traffic_light_status, Table))

        self.assertSetEqual(
            self.db.cam_channels, {"CAM_R2", "CAM_R1", "CAM_R0", "CAM_F0", "CAM_L2", "CAM_L1", "CAM_B0", "CAM_L0"}
        )
        self.assertSetEqual(self.db.lidar_channels, {"MergedPointCloud"})

    def test_nuplan_memory_usage(self) -> None:
        """
        Test that repeatedly creating and destroying nuplan DB objects does not cause memory leaks.
        """

        def spin_up_db() -> None:
            db = get_test_nuplan_db_nocache()
            db.remove_ref()

        starting_usage = 0
        ending_usage = 0
        num_iterations = 5

        hpy = guppy.hpy()
        hpy.setrelheap()

        for i in range(0, num_iterations, 1):
            # Use nested function to ensure local handles go out of scope
            spin_up_db()
            gc.collect()

            heap = hpy.heap()

            # Force heapy to materialize the heap statistics
            # This is done lasily, which can lead to noise if not forced.
            _ = heap.size

            # Skip the first few iterations - there can be noise as caches fill up
            if i == num_iterations - 2:
                starting_usage = heap.size
            if i == num_iterations - 1:
                ending_usage = heap.size

        memory_difference_in_mb = (ending_usage - starting_usage) / (1024 * 1024)

        # Alert on either 100 kb growth or 10 % of starting usage, whichever is bigger
        max_allowable_growth_mb = max(0.1, 0.1 * starting_usage / (1024 * 1024))
        self.assertGreater(max_allowable_growth_mb, memory_difference_in_mb)


if __name__ == "__main__":
    unittest.main()
