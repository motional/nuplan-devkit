import unittest
from unittest.mock import Mock, patch

from nuplan.submission.validators.base_submission_validator import BaseSubmissionValidator
from nuplan.submission.validators.image_is_runnable_validator import ImageIsRunnableValidator


class TestImageIsRunnableValidator(unittest.TestCase):
    """Tests for the ImageIsRunnableValidator class"""

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.validator = ImageIsRunnableValidator()

    def test_construction(self) -> None:
        """Tests that the variables are initialized correctly."""
        self.assertTrue(isinstance(self.validator, BaseSubmissionValidator))

    @patch("nuplan.submission.validators.image_is_runnable_validator.SubmissionContainer")
    def test_validate_runnable(self, mock_submission_container: Mock) -> None:
        """Tests that validator calls the next validator when the image is runnable."""
        submission = "foo"

        with patch.object(BaseSubmissionValidator, "validate") as mock_validate:
            self.validator.validate(submission)
            mock_submission_container.return_value.start.assert_called_once()
            mock_validate.assert_called_with(submission)

    @patch("nuplan.submission.validators.image_is_runnable_validator.SubmissionContainer")
    def test_validate_not_runnable(self, mock_submission_container: Mock) -> None:
        """Tests that validator returns False when image is not runnable."""
        mock_submission_container.return_value.wait_until_running.side_effect = TimeoutError

        result = self.validator.validate("foo")
        self.assertFalse(result)
