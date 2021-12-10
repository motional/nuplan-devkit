import os
import pickle
import unittest

from nuplan.database.common.db import Table
from nuplan.database.nuplan_db.nuplandb import NuPlanDB, NuPlanDBExplorer


class TestNuPlanDB(unittest.TestCase):
    """ Test NuPlanDB. """

    def setUp(self) -> None:

        self.db = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'))
        self.db_explorer = NuPlanDBExplorer(self.db)

    def test_pickle(self) -> None:
        """ Test dumping and loading the object through pickle. """

        db_binary = pickle.dumps(self.db)
        re_db: NuPlanDB = pickle.loads(db_binary)

        self.assertEqual(self.db.data_root, re_db.data_root)
        self.assertEqual(self.db.version, re_db.version)
        self.assertEqual(self.db._verbose, re_db._verbose)

    def test_table_getters(self) -> None:
        """ Test the table getters. """

        self.assertTrue(isinstance(self.db.category, Table))
        self.assertTrue(isinstance(self.db.log, Table))
        self.assertTrue(isinstance(self.db.camera, Table))
        self.assertTrue(isinstance(self.db.lidar, Table))
        self.assertTrue(isinstance(self.db.image, Table))
        self.assertTrue(isinstance(self.db.lidar_pc, Table))
        self.assertTrue(isinstance(self.db.lidar_box, Table))
        self.assertTrue(isinstance(self.db.track, Table))
        self.assertTrue(isinstance(self.db.scene, Table))
        self.assertTrue(isinstance(self.db.scenario_tag, Table))
        self.assertTrue(isinstance(self.db.traffic_light_status, Table))

        self.assertSetEqual(self.db.cam_channels, {
            'CAM_R2', 'CAM_R1', 'CAM_R0', 'CAM_F0',
            'CAM_L2', 'CAM_L1', 'CAM_B0', 'CAM_L0'})
        self.assertSetEqual(self.db.lidar_channels, {'MergedPointCloud'})

    def test_list_categories(self) -> None:
        """ Test list categories. """

        self.db.list_categories()


if __name__ == '__main__':
    unittest.main()
