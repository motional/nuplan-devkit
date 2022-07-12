import unittest
from unittest.mock import Mock, patch

from nuplan.submission.submission_container_manager import SubmissionContainerManager


class TestSubmissionContainerManager(unittest.TestCase):
    """Tests for SubmissionContainerManager class"""

    @patch("nuplan.submission.submission_container_manager.SubmissionContainerFactory")
    def setUp(self, mock_container_factory: Mock) -> None:
        """Sets variables for testing"""
        self.container_manager = SubmissionContainerManager(mock_container_factory)

    @patch("nuplan.submission.submission_container_manager.SubmissionContainerFactory")
    def test_initialization(self, mock_container_factory: Mock) -> None:
        """Tests that objects are initialized correctly."""
        submission_container_manager = SubmissionContainerManager(mock_container_factory)

        self.assertEqual(mock_container_factory, submission_container_manager.submission_container_factory)
        self.assertEqual({}, submission_container_manager.submission_containers)

    def test_get_submission_container(self) -> None:
        """Tests that maps are retrieved from cache, if not present created and added to it."""
        image_name = "image_name"
        container_name = "container_name"
        port = 123
        self.container_manager.submission_container_factory.build_submission_container.return_value = "container"

        _container = self.container_manager.get_submission_container(image_name, container_name, port)

        # We expect the container to be built and to be in the dict
        self.container_manager.submission_container_factory.build_submission_container.assert_called_once_with(
            image_name, container_name, port
        )

        self.assertTrue(container_name in self.container_manager.submission_containers)
        self.assertEqual("container", _container)

        # If we call the get map again, we expect it to be cached
        _ = self.container_manager.get_submission_container(image_name, container_name, port)
        self.container_manager.submission_container_factory.build_submission_container.assert_called_once_with(
            image_name, container_name, port
        )


if __name__ == '__main__':
    unittest.main()
