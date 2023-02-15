import os
import unittest
from typing import Any, Dict, List
from unittest.mock import Mock, call, patch

from nuplan.common.utils.helpers import get_unique_incremental_track_id, get_unique_job_id, keep_trying, try_n_times


class HelperTestingSetup:
    """Helper configuration class for testing"""

    def __init__(self) -> None:
        """Initializes with mock values"""
        self.args: List[Any] = list()
        self.kwargs: Dict[str, Any] = dict()
        self.errors = (RuntimeError,)

        self.passing_function = Mock(return_value="result")
        self.failing_function = Mock(return_value="result", side_effect=self.errors[0])


class TestTryNTimes(unittest.TestCase, HelperTestingSetup):
    """Test suite for tests that lets tests run multiple times before declaring failure."""

    def setUp(self) -> None:
        """Inherited, see superclass"""
        HelperTestingSetup.__init__(self)

    def test_fails_on_invalid_number_of_tries(self) -> None:
        """Tests that we calling this method with zero tries result in failure."""
        with self.assertRaises(AssertionError):
            _ = try_n_times(self.passing_function, [], {}, self.errors, max_tries=0)

    def test_pass_on_valid_cases(self) -> None:
        """Tests that for nominal cases the output of the function is returned."""
        result = try_n_times(self.passing_function, self.args, self.kwargs, self.errors, max_tries=1)
        self.assertEqual("result", result)
        self.passing_function.assert_called_once_with(*self.args, **self.kwargs)

    @patch("time.sleep")
    def test_fail_on_invalid_case_after_n_tries(self, mock_sleep: Mock) -> None:
        """Tests that the helper throws after too many attempts."""
        with self.assertRaises(self.errors[0]):
            _ = try_n_times(self.failing_function, self.args, self.kwargs, self.errors, max_tries=2, sleep_time=4.2)
        calls = [call(*self.args, **self.kwargs)] * 2
        self.failing_function.assert_has_calls(calls)
        mock_sleep.assert_called_with(4.2)


class TestKeepTrying(unittest.TestCase, HelperTestingSetup):
    """Test suite for tests that lets tests run until a timeout is reached before declaring failure."""

    def setUp(self) -> None:
        """Inherited, see superclass"""
        HelperTestingSetup.__init__(self)

    def test_fails_on_invalid_number_of_tries(self) -> None:
        """Tests that we calling this method with zero tries result in failure."""
        with self.assertRaises(AssertionError):
            _ = keep_trying(self.passing_function, [], {}, self.errors, timeout=0.0)

    def test_pass_on_valid_cases(self) -> None:
        """Tests that for nominal cases the output of the function is returned."""
        result, _ = keep_trying(self.passing_function, self.args, self.kwargs, self.errors, timeout=1)
        self.assertEqual("result", result)
        self.passing_function.assert_called_once_with(*self.args, **self.kwargs)

    def test_fail_on_invalid_case_after_timeout(self) -> None:
        """Tests that the helper throws after timeout."""
        with self.assertRaises(TimeoutError):
            _ = keep_trying(self.failing_function, self.args, self.kwargs, self.errors, timeout=1e-6, sleep_time=1e-5)
        self.failing_function.assert_called_with(*self.args, **self.kwargs)


class TestGetUniqueJobId(unittest.TestCase):
    """Test suite for the generation of unique job IDs"""

    def test_unique_job_id_no_job_id(self) -> None:
        """
        Tests that if NUPLAN_JOB_ID is not set, the same job ID is returned.
        """
        if os.environ.get("NUPLAN_JOB_ID") is not None:
            del os.environ['NUPLAN_JOB_ID']

        get_unique_job_id.cache_clear()
        job_id_1 = get_unique_job_id()
        job_id_2 = get_unique_job_id()
        self.assertEqual(job_id_1, job_id_2)

    def test_unique_job_id_with_job_id(self) -> None:
        """
        Tests that if NUPLAN_JOB_ID is not set, the same job ID is returned.
        """
        os.environ["NUPLAN_JOB_ID"] = "12345"
        get_unique_job_id.cache_clear()
        job_id_1 = get_unique_job_id()
        job_id_2 = get_unique_job_id()
        self.assertEqual(job_id_1, job_id_2)

        del os.environ['NUPLAN_JOB_ID']

    def test_uniqueness_job_id(self) -> None:
        """
        Tests that the returned job ids are unique if they are not cached.
        """
        if os.environ.get("NUPLAN_JOB_ID") is not None:
            del os.environ['NUPLAN_JOB_ID']

        job_id_1 = get_unique_job_id()
        get_unique_job_id.cache_clear()
        job_id_2 = get_unique_job_id()
        self.assertNotEqual(job_id_1, job_id_2)

    def test_get_unique_incremental_track_id(self) -> None:
        """Tests creation of unique track ids."""
        track_token_and_expected_id_0 = ("track_0", 0)
        track_token_and_expected_id_1 = ("track_1", 1)

        self.assertEqual(
            get_unique_incremental_track_id(track_token_and_expected_id_0[0]), track_token_and_expected_id_0[1]
        )

        self.assertEqual(
            get_unique_incremental_track_id(track_token_and_expected_id_1[0]), track_token_and_expected_id_1[1]
        )

        self.assertEqual(
            get_unique_incremental_track_id(track_token_and_expected_id_0[0]), track_token_and_expected_id_0[1]
        )


if __name__ == '__main__':
    unittest.main()
