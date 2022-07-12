import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from nuplan.submission.submission_planner import SubmissionPlanner


class TestSubmissionPlanner(TestCase):
    """Tests for SubmissionPlanner class"""

    @patch("os.getenv", MagicMock())
    @patch("grpc.server", Mock())
    @patch("nuplan.submission.submission_planner.chpb_grpc", Mock())
    @patch("nuplan.submission.submission_planner.MapManager", Mock())
    @patch("nuplan.submission.submission_planner.NuPlanMapFactory", Mock())
    @patch("nuplan.submission.submission_planner.GPKGMapsDB", Mock())
    @patch("nuplan.submission.challenge_servicers.DetectionTracksChallengeServicer", Mock())
    def setUp(self) -> None:
        """Sets variables for testing"""
        mock_planner = Mock()

        self.submission_planner = SubmissionPlanner(mock_planner)

    @patch("os.getenv")
    @patch("grpc.server")
    @patch("nuplan.submission.submission_planner.chpb_grpc")
    @patch("nuplan.submission.submission_planner.MapManager", Mock())
    @patch("nuplan.submission.submission_planner.NuPlanMapFactory", Mock())
    @patch("nuplan.submission.submission_planner.GPKGMapsDB", Mock())
    @patch("nuplan.submission.submission_planner.DetectionTracksChallengeServicer")
    def test_initialization(
        self,
        mock_servicer: Mock,
        mock_grpc: Mock,
        mock_server: Mock,
        mock_getenv: Mock,
    ) -> None:
        """Tests that the class is initialized as intended."""
        mock_getenv.return_value = "1234"

        mock_submission_planner = SubmissionPlanner(Mock())

        args = (mock_servicer(), mock_server())
        mock_grpc.add_DetectionTracksChallengeServicer_to_server.assert_called_with(*args)
        mock_submission_planner.server.add_insecure_port.assert_called_with("[::]:1234")

        # Expect to raise on missing port
        mock_getenv.return_value = ""
        with self.assertRaises(RuntimeError):
            _ = SubmissionPlanner(Mock())

    def test_serve(self) -> None:
        """Tests that the submission planner correctly starts the server."""
        self.submission_planner.serve()

        self.submission_planner.server.start.assert_called_once()
        self.submission_planner.server.wait_for_termination.assert_called_once()


if __name__ == '__main__':
    unittest.main()
