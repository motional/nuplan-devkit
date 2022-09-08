import io
import json
import pickle
import tempfile
import unittest
from datetime import datetime, timezone

import boto3
from boto3.exceptions import Boto3Error
from botocore.stub import Stubber

from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.utils.nuboard_cloud_utils import (
    S3FileContent,
    check_s3_nuboard_files,
    download_s3_file,
    download_s3_path,
    get_s3_file_contents,
)


class TestNuBoardCloudUtil(unittest.TestCase):
    """Unit tests for cloud utils in nuboard."""

    def setUp(self) -> None:
        """Set up a list of nuboard files."""
        self.tmp_dir = tempfile.TemporaryDirectory()

    def test_check_s3_nuboard_files_fail(self) -> None:
        """Test if check_s3_nuboard_files fails when there is no nuboard file."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)

        dummy_file_result_message = {
            'dummy_a': S3FileContent(
                filename='dummy_a', size=10, last_modified=datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)
            ),
            'dummy_b': S3FileContent(
                filename='dummy_b', size=10, last_modified=datetime(day=3, month=8, year=1992, tzinfo=timezone.utc)
            ),
        }
        encoded_expected_messages = {
            'dummy_a': json.dumps(dummy_file_result_message['dummy_a'].serialize()).encode(),
            'dummy_b': json.dumps(dummy_file_result_message['dummy_b'].serialize()).encode(),
        }
        dummy_streaming_io_message_response = {}
        expected_params = {}
        for s3_key, result_message in dummy_file_result_message.items():
            dummy_streaming_io_message_response[s3_key] = {'Body': io.BytesIO(encoded_expected_messages[s3_key])}
            expected_params[s3_key] = {'Bucket': 'test-bucket', 'Key': s3_key}

        for s3_key, expected_param in expected_params.items():
            response = dummy_streaming_io_message_response[s3_key]
            stubber.add_response('get_object', response, expected_param)
        with stubber:
            s3_nuboard_file_result_message = check_s3_nuboard_files(
                s3_file_contents=dummy_file_result_message, s3_client=s3_client, s3_path='s3://test-bucket'
            )
        self.assertIsNone(s3_nuboard_file_result_message.nuboard_file)
        self.assertFalse(s3_nuboard_file_result_message.s3_connection_status.success)

    def test_check_s3_nuboard_files_success(self) -> None:
        """Test if check_s3_nuboard_files success when there is a nuboard file."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        nuboard_file = NuBoardFile(
            simulation_main_path=self.tmp_dir.name,
            metric_folder="metrics",
            simulation_folder="simulations",
            metric_main_path=self.tmp_dir.name,
            aggregator_metric_folder="aggregator_metric",
        )
        nuboard_file_name = 'dummy_a' + NuBoardFile.extension()
        dummy_file_result_message = {
            nuboard_file_name: S3FileContent(
                filename=nuboard_file_name,
                size=12,
                last_modified=datetime(day=4, month=5, year=1992, tzinfo=timezone.utc),
            )
        }
        encoded_expected_messages = {
            nuboard_file_name: pickle.dumps(nuboard_file.serialize()),
        }
        dummy_streaming_io_message_response = {}
        expected_params = {}
        for s3_key, result_message in dummy_file_result_message.items():
            dummy_streaming_io_message_response[s3_key] = {'Body': io.BytesIO(encoded_expected_messages[s3_key])}
            expected_params[s3_key] = {'Bucket': 'test-bucket', 'Key': s3_key}
        #
        for s3_key, expected_param in expected_params.items():
            response = dummy_streaming_io_message_response[s3_key]
            stubber.add_response('get_object', response, expected_param)

        with stubber:
            s3_nuboard_file_result_message = check_s3_nuboard_files(
                s3_file_contents=dummy_file_result_message, s3_client=s3_client, s3_path='s3://test-bucket'
            )

        self.assertTrue(s3_nuboard_file_result_message.s3_connection_status.success)
        self.assertIsNotNone(s3_nuboard_file_result_message.nuboard_file)
        self.assertEqual(
            nuboard_file.simulation_main_path, s3_nuboard_file_result_message.nuboard_file.simulation_main_path
        )
        self.assertEqual(nuboard_file.metric_main_path, s3_nuboard_file_result_message.nuboard_file.metric_main_path)

    def test_get_s3_file_content(self) -> None:
        """Test if download_s3_file works."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        expected_response = {
            'CommonPrefixes': [{'Prefix': 'dummy_folder_a/log.txt'}, {'Prefix': 'dummy_folder_b/log_2.txt'}],
            'Contents': [
                {
                    'Key': 'dummy_a',
                    'Size': 15,
                    'LastModified': datetime(day=2, month=7, year=1992, tzinfo=timezone.utc),
                },
                {
                    'Key': 'dummy_b',
                    'Size': 45,
                    'LastModified': datetime(day=6, month=7, year=1992, tzinfo=timezone.utc),
                },
            ],
        }
        expected_params = {'Bucket': 'test-bucket', 'Prefix': '', 'Delimiter': '/'}
        s3_path = 's3://test-bucket'
        stubber.add_response('list_objects_v2', expected_response, expected_params)
        with stubber:
            # There is no .. even include previous folder since s3://test_bucket is the first level of folders
            s3_file_contents = get_s3_file_contents(s3_path=s3_path, client=s3_client, include_previous_folder=True)
            self.assertTrue(s3_file_contents.s3_connection_status.success)
            expected_file_names = ['dummy_folder_a/log.txt', 'dummy_folder_b/log_2.txt', 'dummy_a', 'dummy_b']
            for index, (file_name, _) in enumerate(s3_file_contents.file_contents.items()):
                self.assertEqual(file_name, expected_file_names[index])

    def test_s3_download_file(self) -> None:
        """Test s3_download_file in utils."""
        s3_client = boto3.Session().client('s3')
        dummy_s3_file_content = S3FileContent(
            filename='dummy_a', size=10, last_modified=datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)
        )
        s3_path = 's3://test-bucket'
        save_path = self.tmp_dir.name
        # Expect to raise boto3 error when trying to load since the path is dummy
        with self.assertRaises(Boto3Error):
            download_s3_file(
                s3_path=s3_path, s3_client=s3_client, save_path=save_path, file_content=dummy_s3_file_content
            )

    def test_s3_download_path(self) -> None:
        """Test s3_download_path in utils."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        expected_response = {
            'CommonPrefixes': [{'Prefix': 'dummy_folder_a/log.txt'}, {'Prefix': 'dummy_folder_b/log_2.txt'}],
            'Contents': [
                {
                    'Key': 'dummy_a',
                    'Size': 15,
                    'LastModified': datetime(day=2, month=7, year=1992, tzinfo=timezone.utc),
                },
                {
                    'Key': 'dummy_b',
                    'Size': 45,
                    'LastModified': datetime(day=6, month=7, year=1992, tzinfo=timezone.utc),
                },
            ],
        }
        expected_params = {'Bucket': 'test-bucket', 'Prefix': '', 'Delimiter': '/'}
        s3_path = 's3://test-bucket'
        stubber.add_response('list_objects_v2', expected_response, expected_params)
        save_path = self.tmp_dir.name
        with stubber:
            # Expect to raise boto3 error when trying to load since the path is dummy
            with self.assertRaises(Boto3Error):
                download_s3_path(s3_path=s3_path, s3_client=s3_client, save_path=save_path)

    def tearDown(self) -> None:
        """Remove and clean up the tmp folder."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
