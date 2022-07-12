import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from docker.errors import NotFound as ContainerNotFound

from nuplan.submission.submission_container import SubmissionContainer


class TestSubmissionContainer(TestCase):
    """Tests for SubmissionContainer class"""

    @patch("docker.from_env")
    def setUp(self, mock_from_env: Mock) -> None:
        """Sets variables for testing"""
        self.manager = SubmissionContainer(submission_image="foo/bar", container_name="foo_bar", port=314)

    @patch("docker.from_env", Mock())
    def test_initialization(self) -> None:
        """Tests that the container manager gets initialized correctly."""
        mock_manager = SubmissionContainer(submission_image="foo/bar", container_name="foo_bar", port=314)

        self.assertEqual("foo/bar", mock_manager.submission_image)
        self.assertEqual("foo_bar", mock_manager.container_name)
        self.assertEqual(314, mock_manager.port)

    @patch("docker.from_env")
    @patch.object(SubmissionContainer, 'stop')
    def test_start_submission_container(self, mock_stop_submission_container: Mock, mock_from_env: Mock) -> None:
        """Tests that the container is run with the correct arguments."""
        mock_env = Mock()
        mock_from_env.return_value = mock_env
        mock_env.containers.get.return_value = "test_container"
        test_container = self.manager.start()
        mock_stop_submission_container.assert_called_once()
        self.manager.client.containers.run.assert_called_with(
            'foo/bar',
            name='foo_bar',
            detach=True,
            ports={'314': 314},
            tty=True,
            environment={'SUBMISSION_CONTAINER_PORT': '314'},
            device_requests=[{'Driver': '', 'Count': 0, 'DeviceIDs': ['0'], 'Capabilities': [['gpu']], 'Options': {}}],
            cpuset_cpus='0,1',
            volumes={'/data/sets/nuplan': {'bind': '/data/sets/nuplan', 'mode': 'ro'}},
        )
        self.assertEqual("test_container", test_container)

    def test_stop_missing_container(self) -> None:
        """Checks that trying to remove a missing container does not fail (is intended behavior)"""
        # Check trying to stop a non existing container does not fail and does nothing
        mock_container = Mock()
        self.manager.client = Mock()
        self.manager.client.containers.get.side_effect = ContainerNotFound("Container not found")
        self.manager.client.containers.get.return_value = mock_container

        self.manager.stop()

        self.manager.client.containers.get.assert_called_once()
        mock_container.stop.assert_not_called()
        mock_container.remove.assert_not_called()

    def test_stop_existing_container(self) -> None:
        """Checks that if a running container is found, it is stopped and removed."""
        mock_container = Mock()
        self.manager.client = Mock()
        self.manager.client.containers.get.return_value = mock_container

        self.manager.stop()

        self.manager.client.containers.get.assert_called_once()
        mock_container.kill.assert_called_once()
        mock_container.remove.assert_called_once()


if __name__ == '__main__':
    unittest.main()
