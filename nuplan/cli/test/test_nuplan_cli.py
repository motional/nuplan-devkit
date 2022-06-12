import unittest
from unittest.mock import Mock, PropertyMock, patch

from typer.testing import CliRunner

from nuplan.cli.nuplan_cli import cli

runner = CliRunner()


class TestNuPlanCli(unittest.TestCase):
    """
    Test nuplan cli with typer engine
    """

    @patch("nuplan.cli.db_cli.NuPlanDB")
    def test_db_info_info(self, mock_db: Mock) -> None:
        """Test nuplan_cpi.py db info"""
        mock_db.return_value.__str__.return_value = "DB Info"

        result = runner.invoke(cli, ["db", "info"])
        mock_db.return_value.__str__.assert_called_once()
        self.assertEqual(result.exit_code, 0)

    @patch("nuplan.cli.db_cli.NuPlanDB")
    def test_db_cli_duration(self, mock_db: Mock) -> None:
        """Test nuplan_cpi.py db duration"""
        # Let lidar_pc return list of 1000 elements
        mock_lidar_pc = PropertyMock(return_value=[1] * 1000)
        type(mock_db.return_value).lidar_pc = mock_lidar_pc

        # Invoke the tested cli command
        result = runner.invoke(cli, ["db", "duration"])

        # Expectations check
        mock_lidar_pc.assert_called_once()

        self.assertEqual(result.exit_code, 0)
        self.assertTrue("00:00:50" in result.stdout)

    @patch("nuplan.cli.db_cli.NuPlanDB")
    def test_db_cli_log_vehicle(self, mock_db: Mock) -> None:
        """Test nuplan_cpi.py db log-vehicle"""
        log_data = {"logfile": "SomeLog", "vehicle_name": "Voyager", "vehicle_type": "Spaceship"}
        mock_log = Mock(**log_data)
        mock_logs = PropertyMock(return_value=[mock_log])
        type(mock_db.return_value).log = mock_logs

        result = runner.invoke(cli, ["db", "log-vehicle"])

        mock_logs.assert_called_once()
        for data in log_data.values():
            self.assertTrue(data in result.stdout)
        self.assertEqual(result.exit_code, 0)

    @patch("nuplan.cli.db_cli.NuPlanDB")
    def test_db_cli_scenarios(self, mock_db: Mock) -> None:
        """Test nuplan_cpi.py db scenarios"""
        mock_result = Mock()
        mock_result.distinct.return_value.all.return_value = ["A"]
        mock_db.return_value.session.query.return_value = mock_result
        mock_db.return_value.scenario_tag.select_many.return_value = [1, 2, 3]

        result = runner.invoke(cli, ["db", "scenarios"])
        self.assertEqual(result.exit_code, 0)

        self.assertTrue("The available scenario tags from db:" in result.stdout)
        self.assertTrue("A has 3 scenarios" in result.stdout)


if __name__ == '__main__':
    unittest.main()
