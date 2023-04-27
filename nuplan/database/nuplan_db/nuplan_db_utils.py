from dataclasses import dataclass


@dataclass(frozen=True)
class SensorDataSource:
    """
    Class holding parameters for querying db files to extract sensor data.

    For example, for querying lidar data the attributes would be:
    table: lidar_pc
    sensor_table: lidar
    sensor_token_column: lidar_token (this is how the column holding the sensor token is stored in the `table`
    channel: MergedPointCloud
    """

    table: str
    sensor_table: str
    sensor_token_column: str
    channel: str

    def __post_init__(self) -> None:
        """Checks that the tables provided are compatible"""
        if self.table == 'lidar_pc':
            assert (
                self.sensor_table == "lidar"
            ), f"Incompatible sensor_table: {self.sensor_table} for table {self.table}"
        elif self.table == 'image':
            assert (
                self.sensor_table == "camera"
            ), f"Incompatible sensor_table: {self.sensor_table} for table {self.table}"
        else:
            raise ValueError(f"Unknown requested sensor table: {self.table}!")

        assert (
            self.sensor_token_column == f"{self.sensor_table}_token"
        ), f"Incompatible sensor_token_column: {self.sensor_token_column} for sensor_table {self.sensor_table}"


def get_lidarpc_sensor_data() -> SensorDataSource:
    """
    Builds the SensorDataSource for a lidar_pc.
    :return: The query parameters for lidar_pc.
    """
    return SensorDataSource('lidar_pc', 'lidar', 'lidar_token', 'MergedPointCloud')


def get_camera_channel_sensor_data(channel: str) -> SensorDataSource:
    """
    Builds the SensorDataSource for image from a specified channel.
    :param channel: The channel to select.
    :return: The query parameters for image.
    """
    return SensorDataSource('image', 'camera', 'camera_token', channel)
