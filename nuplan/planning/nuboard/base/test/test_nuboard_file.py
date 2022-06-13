import os
import tempfile
import unittest
from pathlib import Path

from nuplan.planning.nuboard.base.data_class import NuBoardFile


class TestNuBoardFile(unittest.TestCase):
    """Test NuBoardFile functionality."""

    def setUp(self) -> None:
        """Set up a nuBoard file class."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(
            simulation_main_path=self.tmp_dir.name,
            metric_main_path=self.tmp_dir.name,
            metric_folder="metrics",
            simulation_folder="simulations",
            aggregator_metric_folder="aggregator_metric",
        )
        self.nuboard_file_name = Path(self.tmp_dir.name) / ("nuboard_file" + self.nuboard_file.extension())

    def test_nuboard_save_and_load_file(self) -> None:
        """Test saving and loading a nuboard file."""
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)

        self.assertTrue(os.path.exists(self.nuboard_file_name))
        self.assertEqual(self.nuboard_file_name.suffix, self.nuboard_file.extension())

        nuboard_file = NuBoardFile.load_nuboard_file(self.nuboard_file_name)
        self.assertEqual(nuboard_file, self.nuboard_file)

    def tearDown(self) -> None:
        """Clean up temporary folder and files."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
