import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from nuplan.planning.script.run_result_processor_aws import _list_subdirs_filtered


class TestRunResultProcessor(unittest.TestCase):
    """Test ResultProcessor script."""

    def test__list_subdirs_filtered(self) -> None:
        """Tests listing of filtered files in subdirectories."""
        expected_found = []
        with TemporaryDirectory() as tmpdir:
            with TemporaryDirectory(dir=tmpdir) as sub_tmpdir:
                file1 = NamedTemporaryFile(dir=sub_tmpdir, suffix='.yes')
                file2 = NamedTemporaryFile(dir=tmpdir, suffix='.yes')
                expected_found.extend([file1.name, file2.name])
                _ = NamedTemporaryFile(dir=tmpdir, suffix='.no')
                _ = NamedTemporaryFile(dir=sub_tmpdir, suffix='.no')

                paths = _list_subdirs_filtered(Path(tmpdir), path_end_filter='.yes')
        self.assertEqual(set(expected_found), set(paths))


if __name__ == '__main__':
    unittest.main()
