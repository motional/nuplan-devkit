import os
import tempfile
import unittest
from pathlib import Path
from typing import List

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.simulation.observation.observation_type import CameraChannel, LidarChannel


class TestNuPlanScenarioIntegration(unittest.TestCase):
    """Integration test cases for nuplan_scenario.py"""

    def test_get_sensors_at_iteration_download(self) -> None:
        """
        Test that get_sensors_at_iteration is able to pull data from s3 correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario = get_test_nuplan_scenario(sensor_root=tmp_dir)
            sensor_path = Path(f"{tmp_dir}/{scenario.log_name}")

            def _get_image_paths() -> List[Path]:
                """:return: The expected path to the test image file."""
                return list(sensor_path.joinpath(f"{CameraChannel.CAM_R0.value}").glob("*.jpg"))

            def _get_pointcloud_paths() -> List[Path]:
                """:return: The expected path to the test pointcloud file."""
                return list(sensor_path.joinpath(f"{LidarChannel.MERGED_PC.value}").glob("*.pcd"))

            self.assertFalse(os.path.exists(sensor_path))

            sensors = scenario.get_sensors_at_iteration(0, [CameraChannel.CAM_R0, LidarChannel.MERGED_PC])
            self.assertIsNotNone(sensors.pointcloud)
            self.assertIsNotNone(sensors.images)

            # Check if files are as expected.
            self.assertTrue(os.path.exists(sensor_path))
            self.assertTrue(os.path.exists(_get_image_paths()[0]))
            self.assertTrue(os.path.exists(_get_pointcloud_paths()[0]))


if __name__ == '__main__':
    unittest.main()
