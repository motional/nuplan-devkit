import os
import unittest
from typing import Any, Dict
from unittest.mock import Mock, call, patch

from nuplan.submission.evalai.evalai_interface import EvalaiInterface


class MockResponse:
    """Mocks response from server."""

    def __init__(self, data: Dict[str, str], status_code: int) -> None:
        """
        :param data: The data for the response.
        :param status_code: The status code of the response.
        """
        self.data = data
        self.status_code = status_code

    def json(self) -> Dict[str, str]:
        """
        Mimics server response.
        :return: Payload
        """
        return self.data

    def raise_for_status(self) -> None:
        """Mock response."""
        pass


def mocked_put_request(**kwargs: Dict[str, Any]) -> MockResponse:
    """Mocks a PUT request that returns the given payload."""
    return MockResponse(kwargs['data'], 200)


class TestEvalaiInterface(unittest.TestCase):
    """Tests interface class to EvalAI api."""

    @patch.dict(os.environ, {"EVALAI_CHALLENGE_PK": "1234", "EVALAI_PERSONAL_AUTH_TOKEN": "authorization_token"})
    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.evalai = EvalaiInterface("bounce_server")

    def test_initialization(self) -> None:
        """Checks that initialization works and fails as expected."""
        self.assertEqual(self.evalai.EVALAI_AUTH_TOKEN, "authorization_token")
        self.assertEqual(self.evalai.CHALLENGE_PK, "1234")
        self.assertEqual(self.evalai.EVALAI_API_SERVER, "bounce_server")

        with patch.dict(os.environ, {"EVALAI_CHALLENGE_PK": ""}):
            with self.assertRaises(AssertionError):
                _ = EvalaiInterface("server")

        with patch.dict(os.environ, {"EVALAI_PERSONAL_AUTH_TOKEN": ""}):
            with self.assertRaises(AssertionError):
                _ = EvalaiInterface("server")

    @patch('requests.request', side_effect=mocked_put_request)
    def test_update_submission_data(self, mock_put: Mock) -> None:
        """Tests update submission with mock server."""
        test_payload = {'test': "payload"}

        response = self.evalai.update_submission_data(test_payload)

        self.assertEqual(response, test_payload)

        expected_call = call(
            method='PUT',
            url='bounce_server/api/jobs/challenge/1234/update_submission/',
            headers={'Authorization': 'Bearer authorization_token'},
            data=test_payload,
        )
        self.assertEqual(response, test_payload)
        self.assertIn(expected_call, mock_put.call_args_list)
        self.assertEqual(len(mock_put.call_args_list), 1)

    def test_fail_on_missing_api(self) -> None:
        """Test failure of url generation on missing api."""
        with self.assertRaises(AssertionError):
            _ = self.evalai._format_url('missing_api')


if __name__ == '__main__':
    unittest.main()
