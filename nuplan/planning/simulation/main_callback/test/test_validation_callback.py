import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from nuplan.planning.simulation.main_callback.validation_callback import ValidationCallback, _validation_succeeded

TEST_FILE = 'nuplan.planning.simulation.main_callback.validation_callback'


class TestValidationCallback(unittest.TestCase):
    """Tests for the ValidationCallback class"""

    def test_initialization(self) -> None:
        """Tests that the object is constructed correctly"""
        callback = ValidationCallback(output_dir='out', validation_dir_name='validation')
        self.assertEqual(str(callback.output_dir), 'out')
        self.assertEqual(callback._validation_dir_name, 'validation')

    def test_on_run_simulation_end(self) -> None:
        """Tests that the correct files are created in the callback."""
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        callback = ValidationCallback(output_dir=tmp_dir.name, validation_dir_name='validation')

        with patch(f'{TEST_FILE}._validation_succeeded', Mock(return_value=False)):
            callback.on_run_simulation_end()
            self.assertTrue(os.path.exists('/'.join([tmp_dir.name, 'validation', 'failed.txt'])))

        with patch(f'{TEST_FILE}._validation_succeeded', Mock(return_value=True)):
            callback.on_run_simulation_end()
            self.assertTrue(os.path.exists('/'.join([tmp_dir.name, 'validation', 'passed.txt'])))

    def test__validation_succeeded(self) -> None:
        """Tests that helper function reads the runners_report file correctly."""
        # Expect false with a missing runner_report
        with patch(f'{TEST_FILE}.pd.read_parquet', Mock(side_effect=FileNotFoundError)):
            result = _validation_succeeded(Mock())
            self.assertFalse(result)

        # Expect false with a failed runner_report
        failed_df = pd.DataFrame({'succeeded': [True, False]})
        with patch(f'{TEST_FILE}.pd.read_parquet', Mock(return_value=failed_df)):
            result = _validation_succeeded(Mock())
            self.assertFalse(result)

        # Expect true with a succeeded runner_report
        passed_df = pd.DataFrame({'succeeded': [True, True]})
        with patch(f'{TEST_FILE}.pd.read_parquet', Mock(return_value=passed_df)):
            result = _validation_succeeded(Mock())
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
