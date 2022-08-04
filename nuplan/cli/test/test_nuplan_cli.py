import unittest
from typing import Callable, Generator, Tuple

import mock
from typer.testing import CliRunner

from nuplan.cli.nuplan_cli import cli
from nuplan.database.nuplan_db.db_description_types import ColumnDescription, DbDescription, TableDescription

runner = CliRunner()


class TestNuPlanCli(unittest.TestCase):
    """
    Test nuplan cli with typer engine
    """

    def _get_ensure_file_downloaded_patch(
        self, expected_data_root: str, expected_remote_path: str
    ) -> Callable[[str, str], str]:
        """
        Get the patch for ensure_file_downloaded.
        """

        def fxn(actual_data_root: str, actual_remote_path: str) -> str:
            """
            The patch for ensure_file_downloaded.
            """
            self.assertEqual(expected_data_root, actual_data_root)
            self.assertEqual(expected_remote_path, actual_remote_path)

            return actual_remote_path

        return fxn

    def test_db_info_info(self) -> None:
        """
        Test nuplan_cli.py db info command.
        """

        def _patch_get_db_description(log_name: str) -> DbDescription:
            """
            A patch for the get_db_description db function.
            """
            self.assertEqual("expected_log_name", log_name)

            return DbDescription(
                tables={
                    "first_table": TableDescription(
                        name="first_table",
                        row_count=123,
                        columns={
                            "first_token": ColumnDescription(
                                column_id=0, name="first_token", data_type="blob", nullable=False, is_primary_key=True
                            ),
                            "first_something": ColumnDescription(
                                column_id=1,
                                name="first_something",
                                data_type="varchar(64)",
                                nullable=True,
                                is_primary_key=False,
                            ),
                        },
                    ),
                    "second_table": TableDescription(
                        name="second_table",
                        row_count=456,
                        columns={
                            "second_token": ColumnDescription(
                                column_id=0, name="token", data_type="blob", nullable=False, is_primary_key=True
                            ),
                            "second_something": ColumnDescription(
                                column_id=1,
                                name="somthing",
                                data_type="varchar(128)",
                                nullable=True,
                                is_primary_key=False,
                            ),
                        },
                    ),
                }
            )

        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch("/data/sets/nuplan", "expected_log_name")

        with mock.patch("nuplan.cli.db_cli.get_db_description", _patch_get_db_description), mock.patch(
            "nuplan.cli.db_cli._ensure_file_downloaded", ensure_file_downloaded_patch
        ):
            result = runner.invoke(cli, ["db", "info", "expected_log_name"])

            self.assertEqual(0, result.exit_code)

            # Check that some strings are present in the output.
            strings_of_interest = [
                "table first_table: 123 rows",
                "table second_table: 456 rows",
                "column first_token: blob not null primary key",
                "column first_something: varchar(64) null",
                "column second_token: blob not null primary key",
                "column second_something: varchar(128) null",
            ]

            result_stdout = result.stdout.lower()
            for string_of_interest in strings_of_interest:
                self.assertTrue(string_of_interest in result_stdout)

    def test_db_cli_duration(self) -> None:
        """
        Test nuplan_cli.py db duration command.
        """

        def _patch_db_duration(log_name: str) -> int:
            """
            A patch for the get_db_duration function.
            """
            self.assertEqual("expected_log_name", log_name)

            return int(125 * 1e6)

        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch("/data/sets/nuplan", "expected_log_name")

        with mock.patch("nuplan.cli.db_cli.get_db_duration_in_us", _patch_db_duration), mock.patch(
            "nuplan.cli.db_cli._ensure_file_downloaded", ensure_file_downloaded_patch
        ):
            result = runner.invoke(cli, ["db", "duration", "expected_log_name"])

            self.assertEqual(0, result.exit_code)
            self.assertTrue("00:02:05" in result.stdout)

    def test_db_cli_log_duration(self) -> None:
        """
        Test nuplan_cli.py db log-duration command.
        """

        def _patch_db_log_duration(log_name: str) -> Generator[Tuple[str, int], None, None]:
            """
            Patch for get_db_log_duration function.
            """
            self.assertEqual("expected_log_name", log_name)
            for i in range(0, 3, 1):
                # Time chosen arbitrarily
                yield (f"log_file_{i}", int((i + 1) * 67 * 1e6))

        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch("/data/sets/nuplan", "expected_log_name")

        with mock.patch("nuplan.cli.db_cli.get_db_log_duration", _patch_db_log_duration), mock.patch(
            "nuplan.cli.db_cli._ensure_file_downloaded", ensure_file_downloaded_patch
        ):
            result = runner.invoke(cli, ["db", "log-duration", "expected_log_name"])

            self.assertEqual(0, result.exit_code)

            strings_of_interest = [
                "log_file_0 is 00:01:07",
                "log_file_1 is 00:02:14",
                "log_file_2 is 00:03:21",
                "3 total logs",
            ]

            for string_of_interest in strings_of_interest:
                self.assertTrue(string_of_interest in result.stdout)

    def test_db_cli_log_vehicle(self) -> None:
        """
        Test nuplan_cli.py log-vehicle command.
        """

        def _patch_db_log_vehicles(log_name: str) -> Generator[Tuple[str, str], None, None]:
            """
            Patch for get_db_log_vehicles function.
            """
            self.assertEqual("expected_log_name", log_name)

            for i in range(0, 3, 1):
                yield (f"log_file_{i}", f"vehicle_{i}")

        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch("/data/sets/nuplan", "expected_log_name")

        with mock.patch("nuplan.cli.db_cli.get_db_log_vehicles", _patch_db_log_vehicles), mock.patch(
            "nuplan.cli.db_cli._ensure_file_downloaded", ensure_file_downloaded_patch
        ):
            result = runner.invoke(cli, ["db", "log-vehicle", "expected_log_name"])

            self.assertEqual(0, result.exit_code)

            for i in range(0, 3, 1):
                self.assertTrue(f"log_file_{i}, vehicle vehicle_{i}" in result.stdout)

    def test_db_cli_scenarios(self) -> None:
        """
        Test db_cli scenarios command.
        """

        def _patch_db_scenario_info(log_name: str) -> Generator[Tuple[str, int], None, None]:
            """
            Patch for get_db_scenario_info
            """
            self.assertEqual("expected_log_name", log_name)

            for i in range(0, 3, 1):
                yield (f"scenario_{i}", i + 5)

        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch("/data/sets/nuplan", "expected_log_name")

        with mock.patch("nuplan.cli.db_cli.get_db_scenario_info", _patch_db_scenario_info), mock.patch(
            "nuplan.cli.db_cli._ensure_file_downloaded", ensure_file_downloaded_patch
        ):
            result = runner.invoke(cli, ["db", "scenarios", "expected_log_name"])

            self.assertEqual(0, result.exit_code)

            strings_of_interest = ["scenario_0: 5", "scenario_1: 6", "scenario_2: 7", "TOTAL: 18"]

            for string_of_interest in strings_of_interest:
                self.assertTrue(string_of_interest in result.stdout)


if __name__ == '__main__':
    unittest.main()
