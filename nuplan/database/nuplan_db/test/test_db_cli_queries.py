import os
import unittest
from pathlib import Path

from nuplan.database.nuplan_db.db_cli_queries import (
    get_db_description,
    get_db_duration_in_us,
    get_db_log_duration,
    get_db_log_vehicles,
    get_db_scenario_info,
)
from nuplan.database.nuplan_db.db_description_types import ColumnDescription
from nuplan.database.nuplan_db.test.minimal_db_test_utils import DBGenerationParameters, generate_minimal_nuplan_db


class TestDbCliQueries(unittest.TestCase):
    """
    Test suite for the DB Cli queries.
    """

    @staticmethod
    def getDBFilePath() -> Path:
        """
        Get the location for the temporary SQLite file used for the test DB.
        :return: The filepath for the test data.
        """
        return Path("/tmp/test_db_cli_queries.sqlite3")

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create the mock DB data.
        """
        db_file_path = TestDbCliQueries.getDBFilePath()
        if db_file_path.exists():
            db_file_path.unlink()

        generation_parameters = DBGenerationParameters(
            num_lidars=1,
            num_cameras=2,
            num_sensor_data_per_sensor=50,
            num_lidarpc_per_image_ratio=2,
            num_scenes=10,
            num_traffic_lights_per_lidar_pc=5,
            num_agents_per_lidar_pc=3,
            num_static_objects_per_lidar_pc=2,
            scene_scenario_tag_mapping={
                5: ["first_tag"],
                6: ["first_tag", "second_tag"],
            },
            file_path=str(db_file_path),
        )

        generate_minimal_nuplan_db(generation_parameters)

    def setUp(self) -> None:
        """
        The method to run before each test.
        """
        self.db_file_name = str(TestDbCliQueries.getDBFilePath())

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Destroy the mock DB data.
        """
        db_file_path = TestDbCliQueries.getDBFilePath()
        if os.path.exists(db_file_path):
            os.remove(db_file_path)

    def test_get_db_description(self) -> None:
        """
        Test the get_db_description queries.
        """
        db_description = get_db_description(self.db_file_name)

        # Check that all expected tables are present
        expected_tables = [
            "category",
            "ego_pose",
            "lidar",
            "lidar_box",
            "lidar_pc",
            "log",
            "scenario_tag",
            "scene",
            "track",
            "traffic_light_status",
            "camera",
            "image",
        ]

        self.assertEqual(len(expected_tables), len(db_description.tables))
        for expected_table in expected_tables:
            self.assertTrue(expected_table in db_description.tables)

        # Pick one table to validate further
        lidar_pc_table = db_description.tables["lidar_pc"]
        self.assertEqual("lidar_pc", lidar_pc_table.name)
        self.assertEqual(50, lidar_pc_table.row_count)
        self.assertEqual(8, len(lidar_pc_table.columns))

        # Sort columns for ease of unit testing.
        columns = sorted(lidar_pc_table.columns.values(), key=lambda x: x.column_id)  # type: ignore

        def _validate_column(
            column: ColumnDescription,
            expected_id: int,
            expected_name: str,
            expected_data_type: str,
            expected_nullable: bool,
            expected_is_primary_key: bool,
        ) -> None:
            """
            A quick method to validate column info to reduce boilerplate.
            """
            self.assertEqual(expected_id, column.column_id)
            self.assertEqual(expected_name, column.name)
            self.assertEqual(expected_data_type, column.data_type)
            self.assertEqual(expected_nullable, column.nullable)
            self.assertEqual(expected_is_primary_key, column.is_primary_key)

        _validate_column(columns[0], 0, "token", "BLOB", False, True)
        _validate_column(columns[1], 1, "next_token", "BLOB", True, False)
        _validate_column(columns[2], 2, "prev_token", "BLOB", True, False)
        _validate_column(columns[3], 3, "ego_pose_token", "BLOB", False, False)
        _validate_column(columns[4], 4, "lidar_token", "BLOB", False, False)
        _validate_column(columns[5], 5, "scene_token", "BLOB", True, False)
        _validate_column(columns[6], 6, "filename", "VARCHAR(128)", True, False)
        _validate_column(columns[7], 7, "timestamp", "INTEGER", True, False)

    def test_get_db_duration_in_us(self) -> None:
        """
        Test the get_db_duration_in_us query
        """
        duration = get_db_duration_in_us(self.db_file_name)
        self.assertEqual(49 * 1e6, duration)

    def test_get_db_log_duration(self) -> None:
        """
        Test the get_db_log_duration query.
        """
        log_durations = list(get_db_log_duration(self.db_file_name))

        self.assertEqual(1, len(log_durations))
        self.assertEqual("logfile", log_durations[0][0])
        self.assertEqual(49 * 1e6, log_durations[0][1])

    def test_get_db_log_vehicles(self) -> None:
        """
        Test the get_db_log_vehicles query.
        """
        log_vehicles = list(get_db_log_vehicles(self.db_file_name))

        self.assertEqual(1, len(log_vehicles))
        self.assertEqual("logfile", log_vehicles[0][0])
        self.assertEqual("vehicle_name", log_vehicles[0][1])

    def test_get_db_scenario_info(self) -> None:
        """
        Test the get_db_scenario_info query.
        """
        scenario_info_tags = list(get_db_scenario_info(self.db_file_name))

        self.assertEqual(2, len(scenario_info_tags))

        self.assertEqual("first_tag", scenario_info_tags[0][0])
        self.assertEqual(2, scenario_info_tags[0][1])

        self.assertEqual("second_tag", scenario_info_tags[1][0])
        self.assertEqual(1, scenario_info_tags[1][1])


if __name__ == "__main__":
    unittest.main()
