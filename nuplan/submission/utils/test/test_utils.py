import tempfile
import unittest

from nuplan.submission.utils.utils import get_submission_logger


class TestUtils(unittest.TestCase):
    """Tests for utils function."""

    def test_submission_logger(self) -> None:
        """Tests the two handlers of the submission logger."""
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        logfile = '/'.join([tmp_dir.name, 'bar.log'])

        logger = get_submission_logger('foo', logfile)

        logger.info("DONT MIND ME")
        logger.warning("HELLO")
        logger.error("WORLD!")

        # Only warning and error should appear on file
        with open(logfile, 'r') as f:
            self.assertEqual(len(f.readlines()), 2)
