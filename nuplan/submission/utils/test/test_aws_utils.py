import os
import unittest
from typing import Any, Dict
from unittest.mock import Mock, call, patch

from nuplan.submission.utils.aws_utils import _create_directories, _download_files, s3_download, s3_download_dir

TEST_FILE_PATH = "nuplan.submission.utils.aws_utils"


class StubS3Client:
    """Helper class for testing with S3 client."""

    def __init__(self, empty: bool) -> None:
        """Creates a mock paginator."""
        self.paginator_results = {
            'InitialToken': {'NextContinuationToken': 'SecondToken', 'Contents': [{'Key': 'prefix/'}]},
            'SecondToken': {'NextContinuationToken': None, 'Contents': [{'Key': 'prefix/file'}]},
        }
        self.empty_paginator = {
            'InitialToken': {'NextContinuationToken': None, 'Contents': None},
        }
        self.empty = empty

    def list_objects_v2(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Function to return a stub paginator."""
        token = kwargs.get('ContinuationToken', 'InitialToken')
        return self.paginator_results[token] if not self.empty else self.empty_paginator[token]  # type: ignore


class TestAWSUtils(unittest.TestCase):
    """Tests for AWS utils."""

    def test__create_directories(self) -> None:
        """Checks the right number of directories is created."""
        local_path_name = 'base_path'
        directories = ['foo', 'bar']

        with patch(f'{TEST_FILE_PATH}.pathlib.Path.mkdir') as mock_mkdir:
            _create_directories(local_path_name, directories)
            self.assertEqual(2, mock_mkdir.call_count)

    @patch(f'{TEST_FILE_PATH}.pathlib.Path.mkdir', Mock)
    def test__download_files(self) -> None:
        """Checks the S3 client is used correctly."""
        mock_client = Mock()
        files = ['file1', 'file2']
        expected_calls = (call('bucket', 'file1', 'dest/file1'), call('bucket', 'file2', 'dest/file2'))

        _download_files('bucket', mock_client, 'dest', files)

        mock_client.download_file.assert_has_calls(expected_calls)

    @patch(f'{TEST_FILE_PATH}._create_directories')
    @patch(f'{TEST_FILE_PATH}._download_files', Mock)
    def test_s3_download_dir_empty(self, mock_dir_create: Mock) -> None:
        """Tests S3 download dir downloads correct targets."""
        mock_client = StubS3Client(empty=True)
        bucket = 'bucket'
        prefix = 'prefix'
        local_path_name = '/my/path'

        s3_download_dir(bucket, mock_client, prefix, local_path_name)

        mock_dir_create.assert_called_once_with(local_path_name, [])

    @patch(f'{TEST_FILE_PATH}._create_directories')
    @patch(f'{TEST_FILE_PATH}._download_files')
    def test_s3_download_dir(self, mock_download: Mock, mock_dir_create: Mock) -> None:
        """Tests S3 download dir downloads correct targets."""
        mock_client = StubS3Client(empty=False)
        bucket = 'bucket'
        prefix = 'prefix'
        local_path_name = '/my/path'

        s3_download_dir(bucket, mock_client, prefix, local_path_name)

        mock_dir_create.assert_called_once_with(local_path_name, ['prefix/'])
        mock_download.assert_called_once_with('bucket', mock_client, local_path_name, ['prefix/file'], None)

    @patch(f'{TEST_FILE_PATH}.boto3.client')
    @patch(f'{TEST_FILE_PATH}.s3_download_dir')
    @patch.dict(
        os.environ,
        {
            "NUPLAN_SERVER_AWS_ACCESS_KEY_ID": "key",
            "NUPLAN_SERVER_AWS_SECRET_ACCESS_KEY": "secret",
            "NUPLAN_SERVER_S3_ROOT_URL": 'bucket',
        },
        clear=True,
    )
    def test_s3_download(self, mock_download_dir: Mock, mock_client: Mock) -> None:
        """Tests S3 download calls the correct api."""
        s3_download('prefix', 'path')

        mock_client.assert_called_once_with(
            's3', aws_access_key_id='key', aws_secret_access_key='secret', region_name='us-east-1'
        )
        mock_download_dir.assert_called_once_with('bucket', mock_client.return_value, 'prefix', 'path', None)


if __name__ == '__main__':
    unittest.main()
