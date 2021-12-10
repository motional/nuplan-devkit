import os
import unittest

from nuplan.database.nuplan_db.nuplandb import NuPlanDB


class TestRendering(unittest.TestCase):
    """
    These tests don't assert anything, but they will fail if the rendering code
    throws an exception.
    """

    def setUp(self) -> None:
        self.nuplandb = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'),
                                 map_version="nuplan-maps-v0.1")

    def test_LidarPc_render(self) -> None:
        """ Test Lidar PC render."""
        self.nuplandb.lidar_pc['1df0eeb62a5f5d10b1218d9a47c4c8c6'].render()


if __name__ == '__main__':
    unittest.main()
