import unittest
from unittest.mock import Mock, patch

from nuplan.submission.validators.base_submission_validator import BaseSubmissionValidator
from nuplan.submission.validators.image_exists_validator import ImageExistsValidator


class TestImageExistsValidator(unittest.TestCase):
    """Tests for the ImageExistsValidator"""

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.validator = ImageExistsValidator()

    def test_construction(self) -> None:
        """Tests that the variables are initialized correctly."""
        self.assertTrue(isinstance(self.validator, BaseSubmissionValidator))

    @patch("docker.from_env")
    def test_validate(self, mock_env: Mock) -> None:
        """Tests that the validator behaves as intended"""
        missing_submission = "foo"
        present_submission = "bar"
        mock_env.return_value.images.list.return_value = ["bar", "b"]

        self.assertEqual(False, self.validator.validate(missing_submission))

        with patch.object(BaseSubmissionValidator, "validate") as mock_validate:
            self.validator.validate(present_submission)
            mock_validate.assert_called_with(present_submission)
