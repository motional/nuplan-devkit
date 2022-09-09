import os
import tempfile
import unittest
from unittest.mock import patch

from nuplan.planning.simulation.main_callback.completion_callback import CompletionCallback

TEST_FILE = 'nuplan.planning.simulation.main_callback.completion_callback'


class TestValidationCallback(unittest.TestCase):
    """Tests for the ValidationCallback class"""

    @patch.dict(os.environ, {"NUPLAN_SERVER_S3_ROOT_URL": "my-bucket"})
    @patch.dict(os.environ, {"SCENARIO_FILTER_ID": "1"})
    def setUp(self) -> None:
        """Sets up callback for testing."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

        self.callback = CompletionCallback(output_dir=self.tmp_dir.name, challenge_name='challenge')

    def test_initialization(self) -> None:
        """Tests initialization of callback."""
        self.assertEqual(str(self.callback._bucket), 'my-bucket')
        self.assertEqual(
            str(self.callback._completion_dir), '/'.join([self.tmp_dir.name, 'simulation-results/challenge_1'])
        )

    @patch.dict(os.environ, {"NUPLAN_SERVER_S3_ROOT_URL": ""})
    def test_fail_on_missing_bucket(self) -> None:
        """Tests that initialization raises when missing the target bucket."""
        with self.assertRaises(AssertionError):
            _ = CompletionCallback(output_dir='out', challenge_name='challenge')

    def test_on_simulation_end_secondary_instance(self) -> None:
        """Tests that the correct files are created in the callback."""
        self.callback.on_run_simulation_end()

        self.assertTrue(os.path.exists(self.callback._completion_dir / 'completed.txt'))


if __name__ == '__main__':
    unittest.main()
