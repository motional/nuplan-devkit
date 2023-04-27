import os.path
import unittest
from pathlib import Path
from typing import BinaryIO
from unittest.mock import Mock, patch

from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.database.nuplan_db.image import Image
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    convert_legacy_nuplan_path_to_latest,
    download_and_cache,
    infer_remote_key_from_local_path,
    load_image,
    load_point_cloud,
)

NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', "/data/sets/nuplan/")
TEST_PATH = "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils"


class TestNuPlanScenarioUtils(unittest.TestCase):
    """Test cases for nuplan_scenario_utils.py"""

    def test_convert_legacy_nuplan_path_to_latest(self) -> None:
        """Test that convert_legacy_nuplan_path_to_latest works as expected."""
        # Nominal case
        legacy_path = Path(NUPLAN_DATA_ROOT) / "nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db"
        legacy_path_str = str(legacy_path)
        expected_latest_path = (
            Path(NUPLAN_DATA_ROOT) / "nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db"
        )
        expected_latest_path_str = str(expected_latest_path)

        # Test without supplying custom nuplan_data_root, should be fine, since we're using NUPLAN_DATA_ROOT
        actual_latest_path = convert_legacy_nuplan_path_to_latest(legacy_path_str)
        self.assertEqual(expected_latest_path_str, actual_latest_path)

        # Test with supplying custom nuplan_data_root
        actual_latest_path = convert_legacy_nuplan_path_to_latest(legacy_path_str, NUPLAN_DATA_ROOT)
        self.assertEqual(expected_latest_path_str, actual_latest_path)

        # Edge case - data root not ending with '/'
        data_root_without_slash = NUPLAN_DATA_ROOT.rstrip("/")

        actual_latest_path = convert_legacy_nuplan_path_to_latest(legacy_path_str, data_root_without_slash)
        self.assertEqual(expected_latest_path_str, actual_latest_path)

    def test_convert_legacy_nuplan_path_to_latest_invalid_path(self) -> None:
        """Test that convert_legacy_nuplan_path_to_latest will throw if path does not contain version info."""
        invalid_legacy_path = Path(NUPLAN_DATA_ROOT) / "mini/2021.09.16.15.12.03_veh-42_01037_01434.db"
        invalid_legacy_path_str = str(invalid_legacy_path)

        with self.assertRaises(ValueError):
            _ = convert_legacy_nuplan_path_to_latest(invalid_legacy_path_str)

    def test_infer_remote_key_from_local_path(self) -> None:
        """Test that infer_remote_key_from_local_path works as expected."""
        # Nominal case
        local_path = Path(NUPLAN_DATA_ROOT) / "nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db"
        local_path_str = str(local_path)
        expected_remote_key = "splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db"

        # Test without supplying custom nuplan_data_root, should be fine, since we're using NUPLAN_DATA_ROOT
        actual_remote_key = infer_remote_key_from_local_path(local_path_str)
        self.assertEqual(expected_remote_key, actual_remote_key)

        # Test with supplying custom nuplan_data_root
        actual_remote_key = infer_remote_key_from_local_path(local_path_str, NUPLAN_DATA_ROOT)
        self.assertEqual(expected_remote_key, actual_remote_key)

        # Edge case - data root not ending with '/'
        data_root_without_slash = NUPLAN_DATA_ROOT.rstrip("/")

        actual_remote_key = infer_remote_key_from_local_path(local_path_str, data_root_without_slash)
        self.assertEqual(expected_remote_key, actual_remote_key)

    @patch(f"{TEST_PATH}.LidarPointCloud.from_buffer")
    @patch(f"{TEST_PATH}.download_and_cache")
    def test_load_point_cloud(self, mock_load_sensor: Mock, mock_from_buffer: Mock) -> None:
        """Test load_point_cloud."""
        # Set up
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.filename = "pcd"
        mock_local_store = Mock(spec=LocalStore)
        mock_remote_store = Mock(spec=S3Store)
        mock_load_sensor.return_value = Mock()

        load_point_cloud(mock_lidar_pc, mock_local_store, mock_remote_store)
        mock_load_sensor.assert_called_with(mock_lidar_pc.filename, mock_local_store, mock_remote_store)
        mock_from_buffer.assert_called_with(mock_load_sensor.return_value, mock_lidar_pc.filename)

    @patch(f"{TEST_PATH}.Image.from_buffer")
    @patch(f"{TEST_PATH}.download_and_cache")
    def test_load_image(self, mock_load_sensor: Mock, mock_from_buffer: Mock) -> None:
        """Test load_point_cloud."""
        # Set up
        mock_image = Mock(spec=Image)
        mock_image.filename_jpg = "image"
        mock_local_store = Mock(spec=LocalStore)
        mock_remote_store = Mock(spec=S3Store)
        mock_load_sensor.return_value = Mock()

        load_image(mock_image, mock_local_store, mock_remote_store)
        mock_load_sensor.assert_called_with(mock_image.filename_jpg, mock_local_store, mock_remote_store)
        mock_from_buffer.assert_called_with(mock_load_sensor.return_value)

    def test_download_and_cache(self) -> None:
        """Test download_and_cache."""
        # Set up
        mock_key = "key"
        mock_image = Mock(spec=Image)
        mock_image.filename_jpg = "image"
        mock_local_store = Mock(spec=LocalStore)
        mock_local_store.exists.side_effect = [True, False, False]
        mock_local_store.get.return_value = Mock(spec=BinaryIO)
        mock_local_store.put = Mock()
        mock_remote_store = Mock(spec=S3Store)
        mock_remote_store.get = Mock(return_value=Mock(spec=BinaryIO))

        blob = download_and_cache(mock_key, mock_local_store, mock_remote_store)
        self.assertEqual(mock_local_store.get.return_value, blob)
        self.assertTrue(isinstance(blob, BinaryIO))

        # The function should raise if no remote store was provided and the data does not already exist locally.
        with self.assertRaises(RuntimeError):
            download_and_cache(mock_key, mock_local_store, None)

        blob = download_and_cache(mock_key, mock_local_store, mock_remote_store)
        mock_remote_store.get.assert_called_with(mock_key)
        mock_local_store.put.assert_called_with(mock_key, mock_remote_store.get.return_value)
        self.assertTrue(isinstance(blob, BinaryIO))


if __name__ == '__main__':
    unittest.main()
