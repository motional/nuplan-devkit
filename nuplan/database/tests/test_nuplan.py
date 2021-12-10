import os
import unittest

from nuplan.database.nuplan_db.nuplandb import NuPlanDB


class TestNuplan(unittest.TestCase):
    def test_nuplan(self) -> None:
        """
        A simple unit test to check that we can load NuPlan.
        """
        nuplan_db = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'))
        self.assertIsNotNone(nuplan_db)


if __name__ == '__main__':
    unittest.main()
