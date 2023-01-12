import tempfile
import unittest
import unittest.mock
from pathlib import Path
from typing import Dict, List

from nuplan.common.utils.file_backed_barrier import FileBackedBarrier


class TestFileBackedBarrier(unittest.TestCase):
    """
    A class to test that the file backed barrier works properly.
    """

    def test_file_backed_barrier_functions_normal_case_local(self) -> None:
        """
        Tests that the file backed barrier functions properly locally.
        """
        sleep_interval_sec = 10
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            current_time = 0.0

            def patch_time_sleep(sleep_time: float) -> None:
                """
                A patch for the time.sleep method.
                :param sleep_time: The time to sleep.
                """
                nonlocal current_time

                self.assertEqual(sleep_interval_sec, sleep_time)
                current_time += sleep_interval_sec

                if current_time >= 30:
                    barrier = FileBackedBarrier(tmp_dir)
                    barrier._register_activity_id_complete("2")

            def patch_time_time() -> float:
                """
                A patch for the time.time method.
                :return: The current time.
                """
                nonlocal current_time
                self.assertLess(current_time, 40)
                return current_time

            with unittest.mock.patch(
                "nuplan.common.utils.file_backed_barrier.time.sleep", patch_time_sleep
            ), unittest.mock.patch("nuplan.common.utils.file_backed_barrier.time.time", patch_time_time):
                barrier = FileBackedBarrier(tmp_dir)
                barrier.wait_barrier("1", {"1", "2"}, timeout_s=None, poll_interval_s=sleep_interval_sec)

    def test_file_backed_barrier_functions_normal_case_s3(self) -> None:
        """
        Tests that the file backed barrier functions properly in s3.
        """
        sleep_interval_sec = 10
        sample_s3_path = "s3://ml-caches/mitchell.spryn/barrier"
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            current_time = 0.0

            def patch_time_sleep(sleep_time: float) -> None:
                """
                A patch for the time.sleep method.
                :param sleep_time: The time to sleep.
                """
                nonlocal current_time

                self.assertEqual(sleep_interval_sec, sleep_time)
                current_time += sleep_interval_sec

                if current_time >= 30:
                    barrier = FileBackedBarrier(tmp_dir)
                    barrier._register_activity_id_complete("2")

            def patch_time_time() -> float:
                """
                A patch for the time.time method.
                :return: The current time.
                """
                nonlocal current_time
                self.assertLess(current_time, 40)
                return current_time

            def patch_get_s3_client() -> unittest.mock.Mock:
                """
                Mocks the get_s3_client method.
                """

                def patch_s3_client_put_object(Body: bytes, Bucket: str, Key: str) -> None:
                    """
                    A patch for the s3 client put object method.
                    :param body: The body passed to the method.
                    :param bucket: The bucket passed to the method.
                    :param key: The key passed to the method.
                    """
                    self.assertTrue(len(Body) > 0)
                    self.assertEqual("ml-caches", Bucket)
                    self.assertEqual("mitchell.spryn/barrier/1", Key)

                    check_file = tmp_dir / "1"
                    self.assertFalse(check_file.exists())

                    # Write temp file to mark that upload happened.
                    with open(check_file, "w") as f:
                        f.write("x")

                def patch_s3_client_list_objects_v2(Bucket: str, Prefix: str) -> Dict[str, List[Dict[str, str]]]:
                    """
                    A patch for the s3 client list objects v2 method.
                    :param Bucket: The bucket passed.
                    :param Prefix: The prefix passed.
                    """
                    self.assertEqual("ml-caches", Bucket)
                    self.assertEqual("mitchell.spryn/barrier/", Prefix)

                    return {
                        "Contents": [
                            {"Key": f"s3://ml-caches/mitchell.spryn/barrier/{p.stem}"}
                            for p in tmp_dir.glob("**/*")
                            if p.is_file()
                        ]
                    }

                mock_client = unittest.mock.Mock()
                mock_client.put_object = patch_s3_client_put_object
                mock_client.list_objects_v2 = patch_s3_client_list_objects_v2

                return mock_client

            with unittest.mock.patch(
                "nuplan.common.utils.file_backed_barrier.time.sleep", patch_time_sleep
            ), unittest.mock.patch(
                "nuplan.common.utils.file_backed_barrier.time.time", patch_time_time
            ), unittest.mock.patch(
                "nuplan.common.utils.file_backed_barrier.get_s3_client", patch_get_s3_client
            ):
                barrier = FileBackedBarrier(Path(sample_s3_path))
                barrier.wait_barrier("1", {"1", "2"}, timeout_s=None, poll_interval_s=sleep_interval_sec)

    def test_file_backed_barrier_timeout(self) -> None:
        """
        Tests that the timeout feature works properly.
        """
        sleep_interval_sec = 10
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            current_time = 0.0

            def patch_time_sleep(sleep_time: float) -> None:
                """
                A patch for the time.sleep method.
                :param sleep_time: The time to sleep.
                """
                nonlocal current_time

                self.assertEqual(sleep_interval_sec, sleep_time)
                current_time += sleep_interval_sec

                # Never write the second task finishing to trigger error.

            def patch_time_time() -> float:
                """
                A patch for the time.time method.
                :return: The current time.
                """
                nonlocal current_time
                return current_time

            with unittest.mock.patch(
                "nuplan.common.utils.file_backed_barrier.time.sleep", patch_time_sleep
            ), unittest.mock.patch("nuplan.common.utils.file_backed_barrier.time.time", patch_time_time):
                barrier = FileBackedBarrier(tmp_dir)
                with self.assertRaises(TimeoutError):
                    barrier.wait_barrier("1", {"1", "2"}, timeout_s=40, poll_interval_s=sleep_interval_sec)


if __name__ == "__main__":
    unittest.main()
