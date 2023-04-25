import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional

from nuplan.common.utils.io_utils import (
    NuPath,
    delete_file,
    list_files_in_directory,
    path_exists,
    read_binary,
    read_pickle,
    read_text,
    safe_path_to_string,
    save_buffer,
    save_object_as_pickle,
    save_text,
)
from nuplan.common.utils.test_utils.patch import patch_with_validation


class TestIoUtils(unittest.TestCase):
    """
    A class to test that the I/O utilities in nuplan_devkit function properly.
    """

    def test_nupath(self) -> None:
        """
        Tests that converting NuPath to strings works properly.
        """
        example_s3_path = NuPath("s3://test-bucket/foo/bar/baz.txt")
        expected_s3_str = "s3://test-bucket/foo/bar/baz.txt"
        actual_s3_str = str(example_s3_path)
        self.assertEqual(expected_s3_str, actual_s3_str)

        example_local_path = NuPath("/foo/bar/baz")
        expected_local_str = "/foo/bar/baz"
        actual_local_str = str(example_local_path)
        self.assertEqual(expected_local_str, actual_local_str)

    def test_safe_path_to_string(self) -> None:
        """
        Tests that converting paths to strings safely works properly.
        """
        example_s3_path = Path("s3://test-bucket/foo/bar/baz.txt")
        expected_s3_str = "s3://test-bucket/foo/bar/baz.txt"
        actual_s3_str = safe_path_to_string(example_s3_path)
        self.assertEqual(expected_s3_str, actual_s3_str)

        example_local_path = Path("/foo/bar/baz")
        expected_local_str = "/foo/bar/baz"
        actual_local_str = safe_path_to_string(example_local_path)
        self.assertEqual(expected_local_str, actual_local_str)

        # Should work the same for strings
        example_s3_str_path = "s3://test-bucket/foo/bar/baz.txt"
        expected_s3_str = "s3://test-bucket/foo/bar/baz.txt"
        actual_s3_str = safe_path_to_string(example_s3_str_path)
        self.assertEqual(expected_s3_str, actual_s3_str)

        example_local_str_path = "/foo/bar/baz"
        expected_local_str = "/foo/bar/baz"
        actual_local_str = safe_path_to_string(example_local_str_path)
        self.assertEqual(expected_local_str, actual_local_str)

    def test_save_buffer_locally(self) -> None:
        """
        Tests that saving a buffer locally works properly.
        """
        expected_buffer = b"test"

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "local_buffer.bin"

            save_buffer(output_file, expected_buffer)

            with open(output_file, "rb") as f:
                reconstructed_buffer = f.read()

            self.assertEqual(expected_buffer, reconstructed_buffer)

    def test_save_buffer_s3(self) -> None:
        """
        Tests that saving a buffer to s3 works properly.
        """
        upload_bucket_name = "ml-caches"
        upload_path = Path("foo/bar/baz.bin")
        uploaded_file_contents: Optional[bytes] = None

        async def patch_upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
            """
            Patch for upload_file_to_s3_async method.
            :param local_path: The passed local_path.
            :param s3_key: The passed s3_key.
            :param s3_bucket: The passed s3_bucket.
            """
            nonlocal uploaded_file_contents
            self.assertEqual(upload_bucket_name, s3_bucket)
            self.assertEqual(upload_path, s3_key)

            with open(local_path, "rb") as f:
                uploaded_file_contents = f.read()

        expected_buffer = b"test"

        with patch_with_validation(
            "nuplan.common.utils.io_utils.upload_file_to_s3_async", patch_upload_file_to_s3_async
        ):
            output_file = Path(f"s3://{upload_bucket_name}") / f"{upload_path}"

            save_buffer(output_file, expected_buffer)

            # For MyPy, include the assert as well.
            self.assertIsNotNone(uploaded_file_contents)
            assert uploaded_file_contents is not None
            self.assertEqual(expected_buffer, uploaded_file_contents)

    def test_save_object_as_pickle_locally(self) -> None:
        """
        Tests that saving a pickled object locally works properly.
        """
        expected_object = {"a": 1, "b": 2}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "local.pkl"

            save_object_as_pickle(output_file, expected_object)

            with open(output_file, "rb") as f:
                reconstructed_object = pickle.load(f)

            self.assertEqual(expected_object, reconstructed_object)

    def test_save_object_as_pickle_s3(self) -> None:
        """
        Tests that saving a pickled object to s3 works properly.
        """
        upload_bucket_name = "ml-caches"
        upload_path = Path("foo/bar/baz.pkl")
        uploaded_file_contents: Optional[bytes] = None

        async def patch_upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
            """
            Patch for upload_file_to_s3_async method.
            :param local_path: The passed local_path.
            :param s3_key: The passed s3_key.
            :param s3_bucket: The passed s3_bucket.
            """
            nonlocal uploaded_file_contents
            self.assertEqual(upload_bucket_name, s3_bucket)
            self.assertEqual(upload_path, s3_key)

            with open(local_path, "rb") as f:
                uploaded_file_contents = f.read()

        expected_object = {"a": 1, "b": 2}

        with patch_with_validation(
            "nuplan.common.utils.io_utils.upload_file_to_s3_async", patch_upload_file_to_s3_async
        ):
            output_file = Path(f"s3://{upload_bucket_name}") / f"{upload_path}"

            save_object_as_pickle(output_file, expected_object)

            # For MyPy, include the assert as well.
            self.assertIsNotNone(uploaded_file_contents)
            assert uploaded_file_contents is not None
            reconstructed_object: Dict[str, int] = pickle.loads(uploaded_file_contents)
            self.assertEqual(expected_object, reconstructed_object)

    def test_save_text_locally(self) -> None:
        """
        Tests that saving a text file locally works properly.
        """
        expected_text = "test_save_text_locally."
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "local.txt"

            save_text(output_file, expected_text)

            with open(output_file, "r") as f:
                reconstructed_text = f.read()

            self.assertEqual(expected_text, reconstructed_text)

    def test_save_text_s3(self) -> None:
        """
        Tests that saving a text file to s3 works properly.
        """
        upload_bucket_name = "ml-caches"
        upload_path = Path("foo/bar/baz.pkl")
        uploaded_file_contents: Optional[str] = None

        async def patch_upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
            """
            Patch for upload_file_to_s3_async method.
            :param local_path: The passed local_path.
            :param s3_key: The passed s3_key.
            :param s3_bucket: The passed s3_bucket.
            """
            nonlocal uploaded_file_contents
            self.assertEqual(upload_bucket_name, s3_bucket)
            self.assertEqual(upload_path, s3_key)

            with open(local_path, "r") as f:
                uploaded_file_contents = f.read()

        expected_text = "test_save_text_s3."

        with patch_with_validation(
            "nuplan.common.utils.io_utils.upload_file_to_s3_async", patch_upload_file_to_s3_async
        ):
            output_file = Path(f"s3://{upload_bucket_name}") / f"{upload_path}"

            save_text(output_file, expected_text)

            self.assertIsNotNone(uploaded_file_contents)
            self.assertEqual(expected_text, uploaded_file_contents)

    def test_read_text_locally(self) -> None:
        """
        Tests that reading a text file locally works properly.
        """
        expected_text = "some expected text."

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "read_text_locally.txt"

            with open(output_file, "w") as f:
                f.write(expected_text)

            reconstructed_text = read_text(output_file)

            self.assertEqual(expected_text, reconstructed_text)

    def test_read_text_from_s3(self) -> None:
        """
        Tests that reading a text file from S3 works properly.
        """
        download_bucket = "ml-caches"
        download_key = "my/file/path.txt"
        expected_text = "some expected text."

        full_filepath = Path(f"s3://{download_bucket}") / download_key

        async def patch_read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
            """
            A patch for the read_binary_file_contents_from_s3_async method.
            :param s3_key: The passed key
            :param s3_bucket: The passed bucket.
            """
            self.assertEqual(Path(download_key), s3_key)
            self.assertEqual(download_bucket, s3_bucket)

            return expected_text.encode("utf-8")

        with patch_with_validation(
            "nuplan.common.utils.io_utils.read_binary_file_contents_from_s3_async",
            patch_read_binary_file_contents_from_s3_async,
        ):
            reconstructed_text = read_text(full_filepath)

            self.assertEqual(expected_text, reconstructed_text)

    def test_read_pickle_locally(self) -> None:
        """
        Tests that reading a pickle file locally works properly.
        """
        expected_obj = {"foo": "bar"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "read_text_locally.txt"

            with open(output_file, "wb") as f:
                f.write(pickle.dumps(expected_obj))

            reconstructed_obj = read_pickle(output_file)

            self.assertEqual(expected_obj, reconstructed_obj)

    def test_read_pickle_from_s3(self) -> None:
        """
        Tests that reading a pickle file from S3 works properly.
        """
        download_bucket = "ml-caches"
        download_key = "my/file/path.txt"
        expected_obj = {"foo": "bar"}

        full_filepath = Path(f"s3://{download_bucket}") / download_key

        async def patch_read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
            """
            A patch for the read_binary_file_contents_from_s3_async method.
            :param s3_key: The passed key
            :param s3_bucket: The passed bucket.
            """
            self.assertEqual(Path(download_key), s3_key)
            self.assertEqual(download_bucket, s3_bucket)

            return pickle.dumps(expected_obj)

        with patch_with_validation(
            "nuplan.common.utils.io_utils.read_binary_file_contents_from_s3_async",
            patch_read_binary_file_contents_from_s3_async,
        ):
            reconstructed_obj = read_pickle(full_filepath)

            self.assertEqual(expected_obj, reconstructed_obj)

    def test_read_binary_locally(self) -> None:
        """
        Tests that reading a binary file locally works properly.
        """
        expected_data = bytes([1, 2, 3, 4, 5])

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "read_text_locally.txt"

            with open(output_file, "wb") as f:
                f.write(expected_data)

            reconstructed_data = read_binary(output_file)

            self.assertEqual(expected_data, reconstructed_data)

    def test_read_binary_from_s3(self) -> None:
        """
        Tests that reading a binary file from S3 works properly.
        """
        download_bucket = "ml-caches"
        download_key = "my/file/path.data"
        expected_data = bytes([1, 2, 3, 4, 5])

        full_filepath = Path(f"s3://{download_bucket}") / download_key

        async def patch_read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
            """
            A patch for the read_binary_file_contents_from_s3_async method.
            :param s3_key: The passed key
            :param s3_bucket: The passed bucket.
            """
            self.assertEqual(Path(download_key), s3_key)
            self.assertEqual(download_bucket, s3_bucket)

            return expected_data

        with patch_with_validation(
            "nuplan.common.utils.io_utils.read_binary_file_contents_from_s3_async",
            patch_read_binary_file_contents_from_s3_async,
        ):
            reconstructed_data = read_binary(full_filepath)

            self.assertEqual(expected_data, reconstructed_data)

    def test_path_exists_locally(self) -> None:
        """
        Tests that path_exists works for local files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            file_to_create = tmp_dir_path / "existing.txt"
            file_to_not_create = tmp_dir_path / "not_existing.txt"

            with open(file_to_create, "w") as f:
                f.write("some irrelevant text.")

            self.assertTrue(path_exists(file_to_create))
            self.assertFalse(path_exists(file_to_not_create))

            self.assertTrue(path_exists(tmp_dir_path, include_directories=True))
            self.assertFalse(path_exists(tmp_dir_path, include_directories=False))

    def test_path_exists_s3(self) -> None:
        """
        Tests that path_exists works for s3 files.
        """
        test_bucket = "ml-caches"
        test_parent_dir = "my/file/that"
        test_existing_file = f"{test_parent_dir}/exists.txt"
        test_non_existing_file = f"{test_parent_dir}/does_not_exist.txt"

        test_dir_path = Path(f"s3://{test_bucket}") / test_parent_dir
        test_existing_path = Path(f"s3://{test_bucket}") / test_existing_file
        test_non_existing_path = Path(f"s3://{test_bucket}") / test_non_existing_file

        async def patch_check_s3_object_exists_async(s3_key: Path, s3_bucket: str) -> bool:
            """
            Patches the check_s3_object_exists_async method.
            :param key: The s3 key to check.
            :param bucket: The s3 bucket to check.
            :return: The mocked return value.
            """
            self.assertEqual(test_bucket, s3_bucket)

            if str(s3_key) == test_existing_file:
                return True
            elif str(s3_key) in [test_non_existing_file, test_parent_dir]:
                return False

            self.fail(f"Unexpected path passed to check_s3_object_exists patch: {s3_key}")

        async def patch_check_s3_path_exists_async(s3_path: str) -> bool:
            """
            Patches the check_s3_object_exists_async method.
            :param s3_path: The s3 path to check.
            :return: The mocked return value.
            """
            if s3_path in [safe_path_to_string(test_existing_path), safe_path_to_string(test_dir_path)]:
                return True
            elif s3_path == safe_path_to_string(test_non_existing_path):
                return False

            self.fail(f"Unexpected path passed to check_s3_path_exists patch: {s3_path}")

        with patch_with_validation(
            "nuplan.common.utils.io_utils.check_s3_object_exists_async", patch_check_s3_object_exists_async
        ), patch_with_validation(
            "nuplan.common.utils.io_utils.check_s3_path_exists_async", patch_check_s3_path_exists_async
        ):
            self.assertTrue(path_exists(test_existing_path))
            self.assertFalse(path_exists(test_non_existing_path))

            self.assertTrue(path_exists(test_dir_path, include_directories=True))
            self.assertFalse(path_exists(test_dir_path, include_directories=False))

    def test_list_files_in_directory_locally(self) -> None:
        """
        Tests that list_files_in_directory works for local files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            self.assertEqual(list_files_in_directory(tmp_dir_path), [])

            test_file_contents = {"a.txt": "test file a.", "b.txt": "test file b."}
            for filename, contents in test_file_contents.items():
                with open(tmp_dir_path / filename, "w") as f:
                    f.write(contents)

            output_files_in_directory = list_files_in_directory(tmp_dir_path)
            self.assertEqual(len(output_files_in_directory), len(test_file_contents))
            for output_filepath in output_files_in_directory:
                self.assertIn(output_filepath.name, test_file_contents)

    def test_list_files_in_directory_s3(self) -> None:
        """
        Tests that list_files_in_directory works for s3.
        """
        test_bucket = "ml-caches"
        test_directory_key = Path("test_dir")
        test_directory_s3_path = Path(f"s3://{test_bucket}/{test_directory_key}")
        test_files_in_s3 = ["a.txt", "b.txt"]

        expected_files = [Path(f"{test_directory_key}/{filename}") for filename in test_files_in_s3]
        expected_s3_paths = [Path(f"s3://{test_bucket}") / filename for filename in expected_files]

        async def patch_list_files_in_s3_directory_async(
            s3_key: Path, s3_bucket: str, filter_suffix: str = ""
        ) -> List[Path]:
            """
            Patches the list_files_in_s3_directory_async method.
            :param key: The s3 key of the directory.
            :param bucket: The s3 bucket of the directory.
            :param filter_suffix: Unused.
            :return: The mocked return value.
            """
            self.assertEqual(test_bucket, s3_bucket)
            self.assertEqual(test_directory_key, s3_key)

            return expected_files

        with patch_with_validation(
            "nuplan.common.utils.io_utils.list_files_in_s3_directory_async", patch_list_files_in_s3_directory_async
        ):
            output_filepaths = list_files_in_directory(test_directory_s3_path)
            self.assertEqual(output_filepaths, expected_s3_paths)

    def test_delete_file_locally(self) -> None:
        """
        Tests that delete_file works for local files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            test_file_contents = {"a.txt": "test file a.", "b.txt": "test file b."}
            test_file_paths = [tmp_dir_path / filename for filename in test_file_contents]
            for filename, contents in test_file_contents.items():
                with open(tmp_dir_path / filename, "w") as f:
                    f.write(contents)

            self.assertEqual(set(tmp_dir_path.iterdir()), set(test_file_paths))

            for filename in test_file_contents:
                filepath = tmp_dir_path / filename
                delete_file(filepath)
                self.assertNotIn(filepath, tmp_dir_path.iterdir())

            self.assertEqual(len(list(tmp_dir_path.iterdir())), 0)

            with self.assertRaises(ValueError):
                delete_file(tmp_dir_path)

    def test_delete_file_s3(self) -> None:
        """
        Tests that delete_file works for s3.
        """
        test_bucket = "ml-caches"
        test_directory_key = Path("test_dir")
        test_directory_s3_path = Path(f"s3://{test_bucket}/{test_directory_key}")
        test_files_in_s3 = {"a.txt", "b.txt"}

        def get_s3_key(filename: str) -> Path:
            """
            Turns a filename into an s3 key.
            """
            return Path(f"{test_directory_key}/{filename}")

        def list_s3_keys() -> List[Path]:
            """
            Lists the keys in s3.
            :return: S3 keys in the mocked test directory.
            """
            return [get_s3_key(filename) for filename in test_files_in_s3]

        async def patch_list_files_in_s3_directory_async(
            s3_key: Path, s3_bucket: str, filter_suffix: str = ""
        ) -> List[Path]:
            """
            Patches the list_files_in_s3_directory_async method.
            :param key: The s3 key of the directory.
            :param bucket: The s3 bucket of the directory.
            :param filter_suffix: Unused.
            :return: The mocked return value.
            """
            self.assertEqual(test_bucket, s3_bucket)
            self.assertEqual(test_directory_key, s3_key)

            return list_s3_keys()

        async def patch_delete_file_from_s3_async(s3_key: Path, s3_bucket: str) -> None:
            """
            Patches the delete_file_from_s3_async method.
            :param s3_key: The s3 key to delete.
            :param s3_bucket: The s3 bucket.
            """
            nonlocal test_files_in_s3

            self.assertEqual(test_bucket, s3_bucket)
            self.assertEqual(test_directory_key, s3_key.parent)
            self.assertIn(s3_key.name, test_files_in_s3)

            test_files_in_s3.remove(s3_key.name)

        with patch_with_validation(
            "nuplan.common.utils.io_utils.list_files_in_s3_directory_async", patch_list_files_in_s3_directory_async
        ), patch_with_validation(
            "nuplan.common.utils.io_utils.delete_file_from_s3_async", patch_delete_file_from_s3_async
        ):
            initial_s3_keys = list_s3_keys()
            for filename in test_files_in_s3:
                self.assertIn(get_s3_key(filename), initial_s3_keys)

            for filename in set(test_files_in_s3):
                s3_path = test_directory_s3_path / filename
                delete_file(s3_path)
                self.assertNotIn(get_s3_key(filename), list_s3_keys())


if __name__ == "__main__":
    unittest.main()
