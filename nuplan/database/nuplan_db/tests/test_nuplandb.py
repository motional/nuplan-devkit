import pickle
import unittest

from nuplan.database.common.db import Table
from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.database.tests.nuplan_db_test_utils import get_test_nuplan_db


class TestNuPlanDB(unittest.TestCase):
    """Test main nuPlan database class."""

    def setUp(self) -> None:
        """Set up test case."""
        self.db = get_test_nuplan_db()

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


if __name__ == "__main__":
    unittest.main()
