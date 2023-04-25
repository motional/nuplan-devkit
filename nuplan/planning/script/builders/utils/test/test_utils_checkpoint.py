import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from nuplan.planning.script.builders.utils.utils_checkpoint import (
    extract_last_checkpoint_from_experiment,
    find_last_checkpoint_in_dir,
)

PATCH_PREFIX = "nuplan.planning.script.builders.utils.utils_checkpoint"


class TestUtilsCheckpoint(unittest.TestCase):
    """Test checkpoint utils methods."""

    def setUp(self) -> None:
        """Setup test attributes."""
        self.group = Path("exp")
        self.experiment_uid = Path("2023.01.01.00.00.00")
        self.experiment = Path("experiment_name/job_name") / self.experiment_uid

    @patch.object(Path, 'exists', autospec=True, return_value=False)
    def test_find_last_checkpoint_in_dir_dir_unavailable(self, path_exists_mock: Mock) -> None:
        """Test 'find_last_checkpoint_in_dir' method when directory does not exist."""
        group_dir = self.group / self.experiment.parent

        # Call method under test
        result = find_last_checkpoint_in_dir(group_dir, self.experiment_uid)

        self.assertIsNone(result)

    @patch.object(Path, "exists", autospec=True, return_value=True)
    @patch.object(Path, "iterdir", autospec=True, return_value=[Path("epoch=0.ckpt"), Path("epoch=1.ckpt")])
    def test_find_last_checkpoint_in_dir(self, path_iterdir_mock: Mock, path_exists_mock: Mock) -> None:
        """Test 'find_last_checkpoint_in_dir' method under typical use case."""
        group_dir = self.group / self.experiment.parent

        # Call method under test
        result = find_last_checkpoint_in_dir(group_dir, self.experiment_uid)

        expected = Path("exp/experiment_name/job_name/2023.01.01.00.00.00/checkpoints/epoch=1.ckpt")
        self.assertEqual(result, expected)

    @patch.object(
        Path,
        "iterdir",
        autospec=True,
        return_value=[Path("2023.01.01.00.00.00"), Path("2023.01.01.00.00.01"), Path("2023.01.01.00.00.02")],
    )
    @patch(f"{PATCH_PREFIX}.find_last_checkpoint_in_dir", autospec=True)
    def test_extract_last_checkpoint_from_experiment(
        self, find_last_checkpoint_in_dir_mock: Mock, path_iterdir_mock: Mock
    ) -> None:
        """Test extract_last_checkpoint_from_experiment method."""
        output_dir = self.group / self.experiment
        date_format = "%Y.%m.%d.%H.%M.%S"

        # Call method under test
        _ = extract_last_checkpoint_from_experiment(output_dir, date_format)

        # Should be called only with most recent date
        calls = [call(Path("exp/experiment_name/job_name"), Path("2023.01.01.00.00.02"))]
        find_last_checkpoint_in_dir_mock.assert_has_calls(calls)


if __name__ == '__main__':
    unittest.main()
