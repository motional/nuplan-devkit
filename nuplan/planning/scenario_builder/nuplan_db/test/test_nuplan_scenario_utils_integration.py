import os.path
import unittest
from pathlib import Path

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import download_file_if_necessary


class TestNuPlanScenarioUtilsIntegration(unittest.TestCase):
    """Test cases for nuplan_scenario_utils.py"""

    def setUp(self) -> None:
        """Will be run before every test."""
        # Test setup
        self.data_root = Path("/data/sets/nuplan/nuplan-v1.1/splits/mini/")
        # The selected file should actually exist remotely, this path is hand-picked:
        self.local_path = self.data_root / "2021.09.16.15.12.03_veh-42_01037_01434.db"

    def test_download_file_if_necessary_local_path(self) -> None:
        """
        Test that download_file_if_necessary works as expected with local path input.
        WARNING: This test will attempt to remove and re-download 2021.09.16.15.12.03_veh-42_01037_01434.db from
                 the local splits folder.
        """
        # Remove the file if it exists
        if os.path.exists(self.local_path):
            os.remove(self.local_path)

        # Double check: the file should initially not exist
        self.assertFalse(os.path.exists(self.local_path))

        # Execute test - the function should download the file since it's now necessary to do so
        download_file_if_necessary(str(self.data_root), str(self.local_path))

        # Now it should exist
        self.assertTrue(os.path.exists(self.local_path))

    def test_download_file_if_necessary_remote_path(self) -> None:
        """
        Test that download_file_if_necessary works as expected.
        WARNING: This test will attempt to remove and re-download 2021.09.16.15.12.03_veh-42_01037_01434.db from
                 the local splits folder.
        """
        # Remove the file if it exists
        if os.path.exists(self.local_path):
            os.remove(self.local_path)

        # Double check: the file should initially not exist
        self.assertFalse(os.path.exists(self.local_path))

        remote_path = "s3://nuplan-production/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db"

        # Execute test - the function should download the file since it's now necessary to do so
        download_file_if_necessary(str(self.data_root), remote_path)

        # Now it should exist
        self.assertTrue(os.path.exists(self.local_path))


if __name__ == '__main__':
    unittest.main()
