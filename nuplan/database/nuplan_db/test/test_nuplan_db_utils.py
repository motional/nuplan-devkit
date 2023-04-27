import unittest

from nuplan.database.nuplan_db.nuplan_db_utils import (
    SensorDataSource,
    get_camera_channel_sensor_data,
    get_lidarpc_sensor_data,
)


class TestSensorDataSource(unittest.TestCase):
    """Tests for the SensorDataSource class."""

    def test_initialization(self) -> None:
        """Tests correct initialization and raising of invalid configuration."""
        with self.assertRaisesRegex(AssertionError, "Incompatible sensor_table: camera for table lidar_pc"):
            SensorDataSource('lidar_pc', 'camera', 'camera_token', '')

        with self.assertRaisesRegex(AssertionError, "Incompatible sensor_table: lidar for table image"):
            SensorDataSource('image', 'lidar', 'lidar_token', '')

        with self.assertRaisesRegex(ValueError, "Unknown requested sensor table: unknown"):
            SensorDataSource('unknown', '', '', '')

        with self.assertRaisesRegex(
            AssertionError, "Incompatible sensor_token_column: lidar_token for sensor_table camera"
        ):
            SensorDataSource('image', 'camera', 'lidar_token', '')

        _ = SensorDataSource('lidar_pc', 'lidar', 'lidar_token', '')
        valid_sensor_data_source = SensorDataSource('image', 'camera', 'camera_token', 'channel')

        self.assertEqual(valid_sensor_data_source.table, "image")
        self.assertEqual(valid_sensor_data_source.sensor_table, "camera")
        self.assertEqual(valid_sensor_data_source.sensor_token_column, "camera_token")
        self.assertEqual(valid_sensor_data_source.channel, "channel")

    def test_get_lidarpc_sensor_data(self) -> None:
        """Tests that utility function builds the correct object."""
        sensor_data = get_lidarpc_sensor_data()
        self.assertEqual(sensor_data.table, "lidar_pc")
        self.assertEqual(sensor_data.sensor_table, "lidar")
        self.assertEqual(sensor_data.sensor_token_column, "lidar_token")
        self.assertEqual(sensor_data.channel, "MergedPointCloud")

    def test_get_camera_channel_sensor_data(self) -> None:
        """Tests that utility function builds the correct object."""
        sensor_data = get_camera_channel_sensor_data('channel')
        self.assertEqual(sensor_data.table, "image")
        self.assertEqual(sensor_data.sensor_table, "camera")
        self.assertEqual(sensor_data.sensor_token_column, "camera_token")
        self.assertEqual(sensor_data.channel, "channel")


if __name__ == '__main__':
    unittest.main()
