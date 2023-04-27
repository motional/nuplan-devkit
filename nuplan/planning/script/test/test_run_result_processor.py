import re
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from nuplan.planning.script.run_result_processor_aws import (
    NUM_INSTANCES_PER_CHALLENGE,
    is_submission_successful,
    list_subdirs_filtered,
)


class TestRunResultProcessor(unittest.TestCase):
    """Test ResultProcessor script."""

    def test_is_submission_successful(self) -> None:
        """Tests that is_submission_successful utility function is working as expected."""
        # Test variables
        challenge_names = ["challenge_1", "challenge_2"]

        # Store temporary fs object references, so they don't get automatically deleted once they get out of context:
        temp_dirs = []
        temp_files = []

        # Test directory structure
        # + root
        # |- rubbish_data
        # |- rubbish_data_2
        # |- ...
        # |-+ challenge_folder_1
        # | |- xxx_completed.txt
        # | -- xxx_completed_not.txt
        # ..
        # |-  challenge_folder_n
        with TemporaryDirectory() as tmpdir:
            for _ in challenge_names:
                temp_files.append(NamedTemporaryFile(dir=tmpdir))  # rubbish data
                sub_tmpdir = TemporaryDirectory(dir=tmpdir)
                temp_dirs.append(sub_tmpdir)
                for instance in range(NUM_INSTANCES_PER_CHALLENGE):
                    temp_files.append(NamedTemporaryFile(dir=sub_tmpdir.name, suffix='_completed.txt'))
                    temp_files.append(NamedTemporaryFile(dir=sub_tmpdir.name, suffix='_completed_not.txt'))  # rubbish

            # Nominal case: test should pass
            self.assertTrue(is_submission_successful(challenge_names, Path(tmpdir)))

            # Failing case #1: Add a rogue *_completed.txt file at tmpdir/submission root to make the next test fail
            extra_completed_at_root = NamedTemporaryFile(dir=tmpdir, suffix='_completed.txt')

            self.assertFalse(is_submission_successful(challenge_names, Path(tmpdir)))

            extra_completed_at_root.close()  # Failing case #1 cleanup

            # Failing case #2: Rogue *_completed.txt files at challenge dirs
            extra_completed_at_challenge_dirs = []
            for sub_tmpdir in temp_dirs:
                extra_completed_at_challenge_dirs.append(
                    NamedTemporaryFile(dir=sub_tmpdir.name, suffix='_completed.txt')
                )

            self.assertFalse(is_submission_successful(challenge_names, Path(tmpdir)))

            for item in extra_completed_at_challenge_dirs:  # Failing case #2 cleanup
                item.close()

            # Final test to ensure cleanups are done correctly
            self.assertTrue(is_submission_successful(challenge_names, Path(tmpdir)))

            # Manually close files to avoid TemporaryFileCloser warnings
            for item in temp_files:
                item.close()

    def test_list_subdirs_filtered(self) -> None:
        """Tests listing of filtered files in subdirectories."""
        expected_found = []
        temporary_files = []
        with TemporaryDirectory() as tmpdir:
            with TemporaryDirectory(dir=tmpdir) as sub_tmpdir:
                file1 = NamedTemporaryFile(dir=sub_tmpdir, suffix='.yes')
                file2 = NamedTemporaryFile(dir=tmpdir, suffix='.yes')
                rubbish1 = NamedTemporaryFile(dir=tmpdir, suffix='.no')
                rubbish2 = NamedTemporaryFile(dir=sub_tmpdir, suffix='.no')

                temporary_files.extend([file1, file2, rubbish1, rubbish2])

                expected_found.extend([file1.name, file2.name])
                paths = list_subdirs_filtered(Path(tmpdir), regex_pattern=re.compile(r'\.yes'))
                self.assertEqual(set(expected_found), set(paths))

                # Manually close files to avoid TemporaryFileCloserWarnings
                for temp_file in temporary_files:
                    temp_file.close()


if __name__ == '__main__':
    unittest.main()
