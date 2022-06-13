import os
import tempfile
import unittest
from pathlib import Path
from typing import List

from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.utils.utils import check_nuboard_file_paths, read_nuboard_file_paths


class TestNuBoardUtils(unittest.TestCase):
    """Unit tests for utils in nuboard."""

    def setUp(self) -> None:
        """Set up a list of nuboard files."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_paths: List[str] = []
        self.nuboard_files: List[NuBoardFile] = []
        for i in range(2):
            main_path = os.path.join(self.tmp_dir.name, str(i))
            nuboard_file = NuBoardFile(
                simulation_main_path=main_path,
                metric_folder="metrics",
                simulation_folder="simulations",
                metric_main_path=main_path,
                aggregator_metric_folder="aggregator_metric",
            )
            nuboard_file_name = os.path.join(main_path, "nuboard_file" + NuBoardFile.extension())
            self.nuboard_files.append(nuboard_file)
            self.nuboard_paths.append(nuboard_file_name)

    def test_check_nuboard_file_paths(self) -> None:
        """Test if check_nuboard_file_paths works."""
        # Expected to raise a run time error since the file is not saved.
        self.assertRaises(RuntimeError, check_nuboard_file_paths, self.nuboard_paths)

        # Save nuboard files
        for nuboard_file, nuboard_path in zip(self.nuboard_files, self.nuboard_paths):
            main_path = Path(nuboard_file.simulation_main_path)
            main_path.mkdir(parents=True, exist_ok=True)
            file = Path(nuboard_path)
            nuboard_file.save_nuboard_file(file)

        nuboard_paths = check_nuboard_file_paths(self.nuboard_paths)
        self.assertEqual(len(nuboard_paths), 2)
        self.assertIsInstance(nuboard_paths, list)

        for nuboard_path_name in nuboard_paths:
            self.assertIsInstance(nuboard_path_name, Path)

        # Check with directory
        nuboard_path_head = [os.path.dirname(nuboard_path) for nuboard_path in self.nuboard_paths]
        nuboard_paths = check_nuboard_file_paths(nuboard_path_head)
        self.assertEqual(len(nuboard_paths), 2)
        self.assertIsInstance(nuboard_paths, list)

        for nuboard_path_name in nuboard_paths:
            self.assertIsInstance(nuboard_path_name, Path)

    def test_read_nuboard_file_paths(self) -> None:
        """Test if read_nuboard_file_paths works."""
        nuboard_paths: List[Path] = []

        # Save nuboard files
        for nuboard_file, nuboard_path in zip(self.nuboard_files, self.nuboard_paths):
            main_path = Path(nuboard_file.simulation_main_path)
            main_path.mkdir(parents=True, exist_ok=True)
            file = Path(nuboard_path)
            nuboard_file.save_nuboard_file(file)
            nuboard_paths.append(file)

        nuboard_files = read_nuboard_file_paths(file_paths=nuboard_paths)
        self.assertEqual(len(nuboard_files), 2)
        for nuboard_file in nuboard_files:
            self.assertIsInstance(nuboard_file, NuBoardFile)

    def tearDown(self) -> None:
        """Remove and clean up the tmp folder."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
