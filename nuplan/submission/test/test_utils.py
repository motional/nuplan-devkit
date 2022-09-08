import unittest
from unittest.mock import Mock, patch

from nuplan.submission.utils.utils import find_free_port_number


class TestUtils(unittest.TestCase):
    """Tests for util functions"""

    @patch("socket.socket")
    def test_find_port(self, mock_socket: Mock) -> None:
        """Test that method uses socket to find a free port, and returns it."""
        mock_socket().getsockname.return_value = [0, "1234"]
        port = find_free_port_number()

        # Check a free port was searched
        mock_socket().bind.assert_called_once_with(("", 0))
        # Check connection was closed
        mock_socket().close.assert_called_once()

        self.assertEqual(1234, port)


if __name__ == '__main__':
    unittest.main()
