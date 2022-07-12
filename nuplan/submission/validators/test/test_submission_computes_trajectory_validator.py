import unittest
from unittest.mock import Mock, patch

from nuplan.submission.validators.submission_computes_trajectory_validator import SubmissionComputesTrajectoryValidator


class TestSubmissionComputesTrajectoryValidator(unittest.TestCase):
    """Tests for SubmissionComputesTrajectoryValidator class"""

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.submission_computes_trajectory_validator = SubmissionComputesTrajectoryValidator()

    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.get_test_nuplan_scenario', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SimulationIteration', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SimulationHistoryBuffer', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.PlannerInput', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SubmissionContainerFactory', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.RemotePlanner')
    def test_report_invalid_case(self, mock_remote_planner: Mock) -> None:
        """Tests that if a planner can't provide a trajectory, the validator fails"""
        submission = "foo"

        mock_remote_planner().compute_trajectory.return_value = []
        valid = self.submission_computes_trajectory_validator.validate(submission)
        mock_remote_planner().compute_trajectory.assert_called()
        self.assertFalse(valid)
        self.assertEqual(
            self.submission_computes_trajectory_validator.failing_validator, SubmissionComputesTrajectoryValidator
        )

    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.get_test_nuplan_scenario', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SimulationIteration', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SimulationHistoryBuffer', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.PlannerInput', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SubmissionContainerFactory', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.RemotePlanner')
    def test_report_valid_case(self, mock_remote_planner: Mock) -> None:
        """Checks that the validator succeeds when the planner computes a trajectory"""
        submission = "foo"

        mock_remote_planner().compute_trajectory.return_value = ["my", "wonderful", "trajectory"]
        valid = self.submission_computes_trajectory_validator.validate(submission)
        mock_remote_planner().compute_trajectory.assert_called()
        self.assertTrue(valid)
        self.assertEqual(self.submission_computes_trajectory_validator.failing_validator, None)
