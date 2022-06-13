import logging
import os
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from nuplan.planning.script.builders.logging_builder import LogHandlerConfig, PathKeywordMatch, TqdmLoggingHandler


class TestTqdmLoggingHandler(unittest.TestCase):
    """Test TqdmLoggingHandler class."""

    def test_emit_normal(self) -> None:
        """Test emit function when no errors."""
        tlh = TqdmLoggingHandler()
        log_record = logging.LogRecord('', logging.NOTSET, '', 0, 'A normal logging message.', args=None, exc_info=None)
        self.assertIsNone(tlh.emit(log_record))

    def test_emit_keyboard_interrupt(self) -> None:
        """Test emit when KeyboardInterrupt exception is raised."""
        tlh = TqdmLoggingHandler()
        log_record = logging.LogRecord(
            '', logging.NOTSET, '', 0, 'An interrupted logging message.', args=None, exc_info=None
        )
        # Inject a side_effect to flush function to simulate keyboard interruption.
        tlh.flush = Mock()
        tlh.flush.side_effect = KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            tlh.emit(log_record)

    def test_emit_other_exception(self) -> None:
        """Test emit when an unexpected error is raised."""
        tlh = TqdmLoggingHandler()
        log_record = logging.LogRecord(
            '', logging.NOTSET, '', 0, 'An error-handled logging message.', args=None, exc_info=None
        )
        # Inject a side_effect to flush function to simulate an unexpected exception.
        tlh.flush = Mock()
        tlh.flush.side_effect = MemoryError
        tlh.handleError = Mock()

        tlh.emit(log_record)

        # Should not throw.
        tlh.handleError.assert_called_once_with(log_record)


class TestLogHandlerConfig(unittest.TestCase):
    """Test LogHandlerConfig class."""

    @patch('os.makedirs')
    def test_init(self, mock_makedirs) -> None:  # type: ignore
        """Test class initialization."""
        # Come up with a non-existing directory name.
        unique_path = str(uuid4())
        unique_dir = os.path.join(unique_path, '')

        handler = LogHandlerConfig('LEVEL', unique_dir, 'REGEX')
        self.assertEqual(handler.level, 'LEVEL')
        self.assertEqual(handler.path, unique_dir)
        self.assertEqual(handler.filter_regexp, 'REGEX')

        mock_makedirs.assert_called_once_with(unique_path)


class TestPathKeywordMatch(unittest.TestCase):
    """Test PathKeywordMatch class."""

    # We only care about the path in the record.
    log_record = logging.LogRecord('', logging.NOTSET, '/my/filtered/path', 0, msg='', args=None, exc_info=None)

    def test_default_filter(self) -> None:
        """Test filtering by default pattern, which means no filter."""
        pkm = PathKeywordMatch()
        self.assertTrue(pkm.filter(self.log_record))

    def test_filter(self) -> None:
        """Test filtering by a custom pattern."""
        pkm = PathKeywordMatch(regexp='filtered')
        self.assertFalse(pkm.filter(self.log_record))


if __name__ == '__main__':
    unittest.main()
