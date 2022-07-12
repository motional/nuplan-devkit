import unittest
from unittest.mock import Mock, patch

from nuplan.submission.submission_container_factory import SubmissionContainerFactory


class TestSubmissionContainerFactory(unittest.TestCase):
    """Tests for SubmissionContainerFactory class"""

    @patch("nuplan.submission.submission_container_factory.SubmissionContainer")
    def test_build_submission_container(self, mock_submission_container: Mock) -> None:
        """Tests that the submission container factory correctly calls the builder."""
        submission_image = "foo"
        container_name = "bar"
        port = 1234
        _ = SubmissionContainerFactory.build_submission_container(submission_image, container_name, port)
        mock_submission_container.assert_called_with(submission_image, container_name, port)


if __name__ == '__main__':
    unittest.main()
