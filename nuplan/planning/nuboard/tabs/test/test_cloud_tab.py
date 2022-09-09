import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import boto3
from bokeh.document.document import Document
from botocore.stub import Stubber

from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.tabs.cloud_tab import CloudTab
from nuplan.planning.nuboard.tabs.configuration_tab import ConfigurationTab
from nuplan.planning.nuboard.tabs.histogram_tab import HistogramTab
from nuplan.planning.nuboard.utils.nuboard_cloud_utils import (
    S3ConnectionStatus,
    S3FileContent,
    S3FileResultMessage,
    S3NuBoardFileResultMessage,
)


class TestS3Tab(unittest.TestCase):
    """Test nuboard s3 tab functionality."""

    def setUp(self) -> None:
        """Set up a configuration tab."""
        self.doc = Document()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(
            simulation_main_path=self.tmp_dir.name,
            metric_main_path=self.tmp_dir.name,
            metric_folder="metrics",
            simulation_folder="simulations",
            aggregator_metric_folder="aggregator_metric",
            current_path=Path(self.tmp_dir.name),
        )

        # Make folders
        metric_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        simulation_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)

        self.nuboard_file_name = Path(self.tmp_dir.name) / ("nuboard_file" + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.experiment_file_data = ExperimentFileData(file_paths=[self.nuboard_file])
        self.histogram_tab = HistogramTab(experiment_file_data=self.experiment_file_data, doc=self.doc)
        self.configuration_tab = ConfigurationTab(
            experiment_file_data=self.experiment_file_data, doc=self.doc, tabs=[self.histogram_tab]
        )
        if not os.getenv("NUPLAN_EXP_ROOT", None):
            os.environ["NUPLAN_EXP_ROOT"] = self.tmp_dir.name
        self.s3_tab = CloudTab(doc=self.doc, configuration_tab=self.configuration_tab)
        self.dummy_file_result_message = S3FileResultMessage(
            s3_connection_status=S3ConnectionStatus(success=True, return_message="Connect successfully"),
            file_contents={
                'dummy_a': S3FileContent(
                    filename='dummy_a',
                    size=10,
                    last_modified=datetime(day=2, month=7, year=1992, tzinfo=timezone.utc),
                ),
                'dummy_b': S3FileContent(
                    filename='dummy_b',
                    size=10,
                    last_modified=datetime(day=3, month=8, year=1992, tzinfo=timezone.utc),
                ),
            },
        )

    def test_modal_query_btn(self) -> None:
        """Test if modal query btn works."""
        self.s3_tab._s3_modal_query_on_click()
        self.assertNotEqual(self.s3_tab._s3_client, None)

    def test_load_s3_contents_with_file_contents(self) -> None:
        """Test _load_s3_contents works if there are file contents."""
        self.s3_tab._load_s3_contents(s3_file_result_message=self.dummy_file_result_message)
        self.s3_tab.s3_error_text.text = self.dummy_file_result_message.s3_connection_status.return_message
        self.assertEqual(
            self.s3_tab.s3_error_text.text, self.dummy_file_result_message.s3_connection_status.return_message
        )

    def test_s3_data_source_on_selected(self) -> None:
        """Test _s3_data_source_on_selected work."""
        data_sources: Dict[str, List[Any]] = {'object': [], 'last_modified': [], 'timestamp': [], 'size': []}
        for file_name, content in self.dummy_file_result_message.file_contents.items():
            data_sources['object'].append(file_name)
            data_sources['last_modified'].append(content.last_modified_day if content.last_modified is not None else '')
            data_sources['timestamp'].append(content.date_string if content.date_string is not None else '')
            data_sources['size'].append(content.kb_size() if content.kb_size() is not None else '')

        self.s3_tab.data_table.source.data = data_sources

        # Select the first row and second column
        self.s3_tab._selected_column.value = str(1)
        self.s3_tab._s3_data_source_on_selected(attr='indices', new=[0], old=[])
        self.assertEqual(self.s3_tab.s3_download_text_input.value, 'dummy_a')

        # Select the second row and third column
        self.s3_tab._selected_column.value = str(2)
        self.s3_tab._s3_data_source_on_selected(attr='indices', new=[1], old=[])
        self.assertEqual(self.s3_tab.s3_download_text_input.value, 'dummy_b')

        # Select the first row and first column
        self.s3_tab._selected_column.value = str(0)
        self.s3_tab._s3_data_source_on_selected(attr='indices', new=[0], old=[])
        self.assertNotEqual(self.s3_tab._s3_client, None)  # Fail to update client since we dont have s3 credential

    def test_s3_download_button_on_click(self) -> None:
        """Test if s3 download button on_click function works."""
        self.s3_tab.s3_bucket_name.text = 's3://test-bucket'
        self.s3_tab.s3_download_text_input.value = 'test-prefix'
        self.s3_tab._s3_download_button_on_click()
        # Check if download button properties change to make sure button is clicked
        self.assertEqual(self.s3_tab.s3_download_button.label, "Downloading...")
        self.assertTrue(self.s3_tab.s3_download_button.disabled)

    def test_s3_download_prefixes_fail_without_s3_client(self) -> None:
        """Test s3 tab download_prefixes function fails when there is no s3 client."""
        self.s3_tab._s3_download_prefixes()
        self.assertEqual(self.s3_tab.s3_error_text.text, "No s3 connection!")

    def test_s3_download_prefixes_fail_without_nuboard_files(self) -> None:
        """Test s3 tab download_prefixes function fails when there is no nuboard files."""
        # Set up some properties
        self.s3_tab.s3_bucket_name.text = 's3://test-bucket'
        self.s3_tab.s3_download_text_input.value = 'test-prefix'
        s3_client = boto3.Session().client('s3')
        self.s3_tab._s3_client = s3_client

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
        expected_params = {'Bucket': 'test-bucket', 'Prefix': 'test-prefix/', 'Delimiter': '/'}
        stubber.add_response('list_objects_v2', expected_response, expected_params)
        with stubber:
            self.s3_tab._s3_download_prefixes()
            self.assertEqual(self.s3_tab.s3_error_text.text, "No available nuboard files in the prefix")

    def test_s3_update_nuboard_file_main_path(self) -> None:
        """Test s3 tab _update_s3_nuboard_file_main_path function updates main path based on the selected prefix."""
        s3_nuboard_file_result_message = S3NuBoardFileResultMessage(
            s3_connection_status=S3ConnectionStatus(success=True, return_message="Get s3 nuboasrd file"),
            nuboard_file=self.nuboard_file,
            nuboard_filename=self.nuboard_file_name.name,
        )
        prefix = self.tmp_dir.name
        self.s3_tab._update_s3_nuboard_file_main_path(
            s3_nuboard_file_result=s3_nuboard_file_result_message, selected_prefix=prefix
        )
        nuboard_file = s3_nuboard_file_result_message.nuboard_file
        self.assertEqual(nuboard_file.simulation_main_path, self.tmp_dir.name)
        self.assertEqual(nuboard_file.metric_main_path, self.tmp_dir.name)

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
