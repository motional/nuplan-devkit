import asyncio
import glob
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from nuplan.common.utils.s3_utils import (
    check_s3_object_exists,
    check_s3_path_exists,
    delete_file_from_s3,
    download_directory_from_s3,
    download_file_from_s3,
    get_async_s3_session,
    get_cache_metadata_paths,
    is_s3_path,
    list_files_in_s3_directory,
    read_binary_file_contents_from_s3,
    read_text_file_contents_from_s3,
    split_s3_path,
    upload_file_to_s3,
)
from nuplan.common.utils.test_utils.mock_s3_utils import create_mock_bucket, mock_async_s3, setup_mock_s3_directory


class TestS3Utils(unittest.TestCase):
    """
    A class to test that the S3 utilities function properly.
    """

    def test_is_s3_path(self) -> None:
        """
        Tests that the is_s3_path method works properly.
        """
        self.assertTrue(is_s3_path(Path("s3://foo/bar/baz.txt")))
        self.assertFalse(is_s3_path(Path("/foo/bar/baz")))
        self.assertFalse(is_s3_path(Path("foo/bar/baz")))

        self.assertTrue(is_s3_path("s3://foo/bar/baz.txt"))
        self.assertFalse(is_s3_path("/foo/bar/baz"))
        self.assertFalse(is_s3_path("foo/bar/baz"))

    def test_split_s3_path(self) -> None:
        """
        Tests that the split_s3_path method works properly.
        """
        sample_s3_path = Path("s3://test-bucket/foo/bar/baz.txt")
        expected_bucket = "test-bucket"
        expected_path = Path("foo/bar/baz.txt")

        actual_bucket, actual_path = split_s3_path(sample_s3_path)

        self.assertEqual(expected_bucket, actual_bucket)
        self.assertEqual(expected_path, actual_path)

    @mock_async_s3()
    def test_get_async_s3_session(self) -> None:
        """
        Tests that getting a session works correctly.
        """
        sess_1 = get_async_s3_session()
        sess_2 = get_async_s3_session()
        self.assertEqual(sess_1, sess_2)

        sess_3 = get_async_s3_session(force_new=True)
        sess_4 = get_async_s3_session()
        self.assertNotEqual(sess_2, sess_3)
        self.assertEqual(sess_3, sess_4)

    @mock_async_s3()
    def test_download_directory_from_s3(self) -> None:
        """
        Tests that the download_directory_from_s3 method works properly while mocking AWS.
        Assumes that upload_file_to_s3_async works (used to setup test directory in mock bucket).
        """
        # Expects the test directory to look like this:
        # .
        # ├── dir1
        # │   ├── file2.txt -> "this is file2."
        # │   └── file3.txt -> "this is file3."
        # └── file1.txt     -> "this is file1."
        #
        test_upload_directory = Path("test_download_directory_from_s3")
        test_bucket_name = "test-bucket"

        expected_relative_path_and_contents = {
            "file1.txt": "this is file1.",
            "dir1/file2.txt": "this is file2.",
            "dir1/file3.txt": "this is file3.",
        }

        asyncio.run(
            setup_mock_s3_directory(expected_relative_path_and_contents, test_upload_directory, test_bucket_name)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            expected_directory_path_and_contents = {
                os.path.join(temp_dir, path): contents for path, contents in expected_relative_path_and_contents.items()
            }

            download_directory_from_s3(temp_dir, test_upload_directory, test_bucket_name)

            all_files = glob.glob(f"{temp_dir}/**/*.txt", recursive=True)

            self.assertEqual(len(all_files), len(expected_directory_path_and_contents))
            for key in expected_directory_path_and_contents:
                self.assertTrue(os.path.exists(key))

                with open(key, "r") as f:
                    actual_text = f.read().strip()

                self.assertEqual(expected_directory_path_and_contents[key], actual_text)

    @mock_async_s3()
    def test_list_files_in_s3_directory(self) -> None:
        """
        Tests that the list_files_in_s3_directory method works properly while mocking AWS.
        Assumes that upload_file_to_s3_async works (used to setup test directory in mock bucket).
        """
        # Expects the test directory to look like this:
        # .
        # ├── dir1
        # │   ├── file2.txt -> "this is file2."
        # │   └── file3.txt -> "this is file3."
        # └── file1.txt     -> "this is file1."
        #
        test_files_directory = Path("test_list_files_in_s3_directory")
        test_bucket_name = "test-bucket"

        expected_relative_path_and_contents = {
            "file1.txt": "this is file1.",
            "dir1/file2.txt": "this is file2.",
            "dir1/file3.txt": "this is file3.",
        }

        asyncio.run(
            setup_mock_s3_directory(expected_relative_path_and_contents, test_files_directory, test_bucket_name)
        )

        expected_files = {test_files_directory / path for path in expected_relative_path_and_contents}
        actual_files = list_files_in_s3_directory(test_files_directory, test_bucket_name)

        self.assertEqual(len(expected_files), len(actual_files))
        for file_path in actual_files:
            self.assertTrue(file_path in expected_files)

    @mock_async_s3()
    def test_check_s3_exist_ops(self) -> None:
        """
        Tests that the check_s3_object_exists and check_s3_path_exists methods functions properly while mocking AWS.
        Assumes that upload_file_to_s3_async works (used to setup test directory in mock bucket).
        """
        # Expects the test directory to look like this:
        # .
        # └── existing.txt     -> "this exists."
        #
        def to_s3_path(key: Path, bucket: str) -> str:
            """
            Returns s3 path string from split path.
            :param key: s3 key.
            :param bucket: s3 bucket.
            :return: Unsplit s3 path.
            """
            return f"s3://{bucket}/{key}"

        test_files_directory = Path("test_check_s3_object_exists")
        test_bucket_name = "test-bucket"

        expected_relative_path_and_contents = {
            "existing.txt": "this exists.",
        }

        asyncio.run(
            setup_mock_s3_directory(expected_relative_path_and_contents, test_files_directory, test_bucket_name)
        )

        existing_key = test_files_directory / "existing.txt"
        non_existing_key = test_files_directory / "does_not_exist.txt"

        self.assertTrue(check_s3_object_exists(existing_key, test_bucket_name))
        self.assertTrue(check_s3_path_exists(to_s3_path(existing_key, test_bucket_name)))
        self.assertFalse(check_s3_object_exists(non_existing_key, test_bucket_name))
        self.assertFalse(check_s3_path_exists(to_s3_path(non_existing_key, test_bucket_name)))

        # check_s3_object_exists should return False for directory paths.
        self.assertFalse(check_s3_object_exists(test_files_directory, test_bucket_name))

        # check_s3_path_exists shoudl return True for directory paths.
        self.assertTrue(check_s3_path_exists(to_s3_path(test_files_directory, test_bucket_name)))

    @mock_async_s3()
    def test_get_cache_metadata_paths(self) -> None:
        """
        Tests that the get_cache_metadata_paths method functions properly while mocking AWS.
        Assumes that upload_file_to_s3_async works (used to setup test directory in mock bucket).
        """
        # Expects the test directory to look like this:
        # .
        # ├── metadata
        # │   ├── file2.csv -> "this is file2."
        # │   └── file3.csv -> "this is file3."
        # └── file1.txt     -> "this is file1."
        #
        test_files_directory = Path("test_get_cache_metadata_paths")
        test_bucket_name = "test-bucket"

        expected_relative_path_and_contents = {
            "file1.csv": "this is file1.",
            "metadata/file2.csv": "this is file2.",
            "metadata/file3.csv": "this is file3.",
        }

        asyncio.run(
            setup_mock_s3_directory(expected_relative_path_and_contents, test_files_directory, test_bucket_name)
        )

        # Check that the metadata files are found
        expected_metadata_files = [
            test_files_directory / "metadata/file2.csv",
            test_files_directory / "metadata/file3.csv",
        ]
        actual_metadata_files = get_cache_metadata_paths(test_files_directory, test_bucket_name)

        self.assertEqual(len(expected_metadata_files), len(actual_metadata_files))
        for s3_path in actual_metadata_files:
            bucket, file_path = split_s3_path(s3_path)
            self.assertTrue(file_path in expected_metadata_files)

        # Check that none are found in a non-existing directory
        non_existing_files = get_cache_metadata_paths(
            test_files_directory, test_bucket_name, metadata_folder="non_existing"
        )
        self.assertEqual(len(non_existing_files), 0)

    @mock_async_s3()
    def test_s3_single_file_ops(self) -> None:
        """
        Tests that the following methods work properly while mocking AWS:
        * Upload file to S3
        * Download file from S3
        * Read file from S3
        * Delete file from S3
        """
        upload_bucket_name = "test-bucket"
        asyncio.run(create_mock_bucket(upload_bucket_name))

        test_id = str(uuid.uuid4())
        upload_bucket_folder = Path("test_upload_file_to_s3")
        upload_bucket_path = upload_bucket_folder / f"{test_id}.txt"
        expected_file_contents = f"A random identifier: {test_id}."

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test upload.
            upload_file_path = Path(os.path.join(temp_dir, "upload.txt"))
            with open(upload_file_path, "w") as f:
                f.write(expected_file_contents)

            upload_file_to_s3(upload_file_path, upload_bucket_path, upload_bucket_name)

            # Test that the file exists.
            self.assertEqual(1, len(list_files_in_s3_directory(upload_bucket_path, upload_bucket_name)))

            # Test that the contents can be read into memory.
            read_file_contents = read_text_file_contents_from_s3(upload_bucket_path, upload_bucket_name)
            self.assertEqual(expected_file_contents, read_file_contents)

            # Also test reading with read_binary_file_contents_from_s3.
            read_binary_contents = read_binary_file_contents_from_s3(upload_bucket_path, upload_bucket_name)
            self.assertEqual(expected_file_contents, read_binary_contents.decode("utf-8"))

            # Test the download functionality.
            download_file_path = Path(os.path.join(temp_dir, "download.txt"))
            download_file_from_s3(download_file_path, upload_bucket_path, upload_bucket_name)

            self.assertTrue(os.path.exists(download_file_path))
            with open(download_file_path, "r") as f:
                downloaded_text = f.read()

            self.assertEqual(expected_file_contents, downloaded_text)

            # Test deletion.
            delete_file_from_s3(upload_bucket_path, upload_bucket_name)
            self.assertEqual(0, len(list_files_in_s3_directory(upload_bucket_path, upload_bucket_name)))


if __name__ == "__main__":
    unittest.main()
