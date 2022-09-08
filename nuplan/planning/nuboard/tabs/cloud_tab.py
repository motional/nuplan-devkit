import hashlib
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from bokeh.document.document import Document
from bokeh.models import Button, ColumnDataSource, DataTable, Div, PasswordInput, TableColumn, TextInput
from boto3.exceptions import Boto3Error

from nuplan.common.utils.s3_utils import get_s3_client
from nuplan.planning.nuboard.tabs.config.cloud_tab_config import (
    S3TabBucketNameConfig,
    S3TabBucketTextInputConfig,
    S3TabDataTableConfig,
    S3TabDownloadButtonConfig,
    S3TabDownloadTextInputConfig,
    S3TabErrorTextConfig,
    S3TabLastModifiedColumnConfig,
    S3TabObjectColumnConfig,
    S3TabS3AccessKeyIDTextInputConfig,
    S3TabS3BucketPrefixTextInputConfig,
    S3TabS3ModalQueryButtonConfig,
    S3TabS3SecretAccessKeyPasswordTextInputConfig,
    S3TabSizeColumnConfig,
    S3TabTimeStampColumnConfig,
)
from nuplan.planning.nuboard.tabs.configuration_tab import ConfigurationTab
from nuplan.planning.nuboard.tabs.js_code.cloud_tab_js_code import (
    S3TabContentDataSourceOnSelected,
    S3TabContentDataSourceOnSelectedLoadingJSCode,
    S3TabDataTableUpdateJSCode,
    S3TabDownloadUpdateJSCode,
    S3TabLoadingJSCode,
)
from nuplan.planning.nuboard.utils.nuboard_cloud_utils import (
    S3FileResultMessage,
    S3NuBoardFileResultMessage,
    check_s3_nuboard_files,
    download_s3_file,
    download_s3_path,
    get_s3_file_contents,
)

logger = logging.getLogger(__name__)


class CloudTab:
    """Cloud tab in nuboard."""

    def __init__(self, doc: Document, configuration_tab: ConfigurationTab, s3_bucket: Optional[str] = ''):
        """
        Cloud tab for remote connection features.
        :param doc: Bokeh HTML document.
        :param configuration_tab: Configuration tab.
        :param s3_bucket: Aws s3 bucket name.
        """
        self._doc = doc
        self._configuration_tab = configuration_tab
        self._nuplan_exp_root = os.getenv('NUPLAN_EXP_ROOT', None)
        assert self._nuplan_exp_root is not None, "Please set environment variable: NUPLAN_EXP_ROOT!"
        download_path = Path(self._nuplan_exp_root)
        download_path.mkdir(parents=True, exist_ok=True)

        # Data source
        self._default_datasource_dict = dict(
            object=['-'],
            last_modified=['-'],
            timestamp=['-'],
            size=['-'],
        )
        self._s3_content_datasource = ColumnDataSource(data=self._default_datasource_dict)
        self._selected_column = TextInput()
        self._selected_row = TextInput()

        self.s3_bucket_name = Div(**S3TabBucketNameConfig.get_config())
        self.s3_bucket_name.js_on_change('text', S3TabDataTableUpdateJSCode.get_js_code())

        self.s3_error_text = Div(**S3TabErrorTextConfig.get_config())
        self.s3_download_text_input = TextInput(**S3TabDownloadTextInputConfig.get_config())

        self.s3_download_button = Button(**S3TabDownloadButtonConfig.get_config())
        self.s3_download_button.on_click(self._s3_download_button_on_click)
        self.s3_download_button.js_on_click(S3TabLoadingJSCode.get_js_code())
        self.s3_download_button.js_on_change('disabled', S3TabDownloadUpdateJSCode.get_js_code())

        self.s3_bucket_text_input = TextInput(**S3TabBucketTextInputConfig.get_config(), value=s3_bucket)
        self.s3_access_key_id_text_input = TextInput(**S3TabS3AccessKeyIDTextInputConfig.get_config())
        self.s3_secret_access_key_password_input = PasswordInput(
            **S3TabS3SecretAccessKeyPasswordTextInputConfig.get_config()
        )
        self.s3_bucket_prefix_text_input = TextInput(**S3TabS3BucketPrefixTextInputConfig.get_config())

        self.s3_modal_query_btn = Button(**S3TabS3ModalQueryButtonConfig.get_config())
        self.s3_modal_query_btn.on_click(self._s3_modal_query_on_click)
        self.s3_modal_query_btn.js_on_click(S3TabLoadingJSCode.get_js_code())
        self._default_columns = [
            TableColumn(**S3TabObjectColumnConfig.get_config()),
            TableColumn(**S3TabLastModifiedColumnConfig.get_config()),
            TableColumn(**S3TabTimeStampColumnConfig.get_config()),
            TableColumn(**S3TabSizeColumnConfig.get_config()),
        ]

        self._s3_content_datasource = ColumnDataSource(data=self._default_datasource_dict)
        self._s3_content_datasource.js_on_change('data', S3TabDataTableUpdateJSCode.get_js_code())
        self._s3_content_datasource.selected.js_on_change(
            'indices',
            S3TabContentDataSourceOnSelected.get_js_code(
                selected_column=self._selected_column, selected_row=self._selected_row
            ),
        )
        self._s3_content_datasource.selected.js_on_change(
            'indices',
            S3TabContentDataSourceOnSelectedLoadingJSCode.get_js_code(
                source=self._s3_content_datasource, selected_column=self._selected_column
            ),
        )
        self._s3_content_datasource.selected.on_change('indices', self._s3_data_source_on_selected)
        self.data_table = DataTable(
            source=self._s3_content_datasource, columns=self._default_columns, **S3TabDataTableConfig.get_config()
        )

        self._s3_client: Optional[boto3.client] = None
        if s3_bucket:
            self._update_blob_store(s3_bucket=s3_bucket, s3_prefix='')

    def _update_blob_store(self, s3_bucket: str, s3_prefix: str = '') -> None:
        """
        :param s3_bucket:
        :param s3_prefix:
        """
        # Update s3 client
        aws_profile_name = bytes(
            self.s3_access_key_id_text_input.value + self.s3_secret_access_key_password_input.value, encoding='utf-8'
        )
        hash_md5 = hashlib.md5(aws_profile_name)
        profile = hash_md5.hexdigest()
        self._s3_client = get_s3_client(
            aws_access_key_id=self.s3_access_key_id_text_input.value,
            aws_secret_access_key=self.s3_secret_access_key_password_input.value,
            profile_name=profile,
        )
        s3_path = os.path.join(s3_bucket, s3_prefix)
        s3_file_result_message = get_s3_file_contents(
            s3_path=s3_path, include_previous_folder=True, client=self._s3_client
        )
        self._load_s3_contents(s3_file_result_message=s3_file_result_message)
        self.s3_error_text.text = s3_file_result_message.s3_connection_status.return_message
        if s3_file_result_message.s3_connection_status.success:
            self.s3_bucket_name.text = s3_bucket

    def _s3_modal_query_on_click(self) -> None:
        """On click function for modal query button."""
        self._update_blob_store(
            s3_bucket=self.s3_bucket_text_input.value, s3_prefix=self.s3_bucket_prefix_text_input.value
        )

    def _s3_data_source_on_selected(self, attr: str, old: List[int], new: List[int]) -> None:
        """Helper function when select a row in data source."""
        # Cancel history so that it works if users click on the same row again
        if not new:
            return
        row_index = new[0]
        self._s3_content_datasource.selected.update(indices=[])

        # If users click on the object column
        column_index = int(self._selected_column.value)
        s3_prefix = self.data_table.source.data['object'][row_index]
        if column_index == 0:
            if not s3_prefix or s3_prefix == '-':
                return
            # Return to previous folder
            if '..' in s3_prefix:
                s3_prefix = Path(s3_prefix).parents[1].name
            self._update_blob_store(s3_bucket=self.s3_bucket_text_input.value, s3_prefix=s3_prefix)
        else:
            if '..' in s3_prefix or '-' == s3_prefix:
                return
            self.s3_download_text_input.value = s3_prefix

    def _update_data_table_source(self, data_sources: Dict[str, List[Any]]) -> None:
        """Update data table source."""
        self.data_table.source.data = data_sources

    def _load_s3_contents(self, s3_file_result_message: S3FileResultMessage) -> None:
        """
        Load s3 contents into a data table.
        :param s3_file_result_message: File content and return messages from s3 connection.
        """
        file_contents = s3_file_result_message.file_contents
        # If return fail or less than 1 files (only '..' is added to the list, which means no files exist)
        if not s3_file_result_message.s3_connection_status.success or len(s3_file_result_message.file_contents) <= 1:
            default_data_sources = self._default_datasource_dict
            self._doc.add_next_tick_callback(partial(self._update_data_table_source, data_sources=default_data_sources))
        else:
            data_sources: Dict[str, List[Any]] = {'object': [], 'last_modified': [], 'timestamp': [], 'size': []}
            for file_name, content in file_contents.items():
                data_sources['object'].append(file_name)
                data_sources['last_modified'].append(
                    content.last_modified_day if content.last_modified is not None else ''
                )
                data_sources['timestamp'].append(content.date_string if content.date_string is not None else '')
                data_sources['size'].append(content.kb_size() if content.kb_size() is not None else '')
            self._doc.add_next_tick_callback(partial(self._update_data_table_source, data_sources=data_sources))

    def _reset_s3_download_button(self) -> None:
        """Reset s3 download button."""
        self.s3_download_button.label = 'Download'
        self.s3_download_button.disabled = False
        self.s3_download_text_input.disabled = False

    def _update_error_text_label(self, text: str) -> None:
        """Update error text message in a sequential manner."""
        self.s3_error_text.text = text

    def _s3_download_prefixes(self) -> None:
        """Download s3 prefixes and update progress in a sequential manner."""
        try:
            start_time = time.perf_counter()
            if not self._s3_client:
                raise Boto3Error("No s3 connection!")
            selected_s3_bucket = str(self.s3_bucket_name.text).strip()
            selected_s3_prefix = str(self.s3_download_text_input.value).strip()
            selected_s3_path = os.path.join(selected_s3_bucket, selected_s3_prefix)

            s3_result_file_contents = get_s3_file_contents(
                s3_path=selected_s3_path, client=self._s3_client, include_previous_folder=False
            )
            s3_nuboard_file_result = check_s3_nuboard_files(
                s3_result_file_contents.file_contents, s3_client=self._s3_client, s3_path=selected_s3_path
            )
            if not s3_nuboard_file_result.s3_connection_status.success:
                raise Boto3Error(s3_nuboard_file_result.s3_connection_status.return_message)

            if not s3_result_file_contents.file_contents:
                raise Boto3Error(f"No objects exist in the path: {selected_s3_path}")

            self._download_s3_file_contents(
                s3_result_file_contents=s3_result_file_contents, selected_s3_bucket=selected_s3_bucket
            )
            self._update_s3_nuboard_file_main_path(
                s3_nuboard_file_result=s3_nuboard_file_result, selected_prefix=selected_s3_prefix
            )
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            successful_message = f'Downloaded to {self._nuplan_exp_root} and took {elapsed_time:.4f} seconds'
            logger.info('Downloaded to {} and took {:.4f} seconds'.format(self._nuplan_exp_root, elapsed_time))
            self._doc.add_next_tick_callback(partial(self._update_error_text_label, text=successful_message))
        except Exception as e:
            logger.info(str(e))
            self.s3_error_text.text = str(e)
        self._doc.add_next_tick_callback(self._reset_s3_download_button)

    def _update_s3_nuboard_file_main_path(
        self, s3_nuboard_file_result: S3NuBoardFileResultMessage, selected_prefix: str
    ) -> None:
        """
        Update nuboard file simulation and metric main path.
        :param s3_nuboard_file_result: S3 nuboard file result.
        :param selected_prefix: Selected prefix on s3.
        """
        nuboard_file = s3_nuboard_file_result.nuboard_file
        nuboard_filename = s3_nuboard_file_result.nuboard_filename
        if not nuboard_file or not nuboard_filename or not self._nuplan_exp_root:
            return
        main_path = Path(self._nuplan_exp_root) / selected_prefix
        nuboard_file.simulation_main_path = str(main_path)
        nuboard_file.metric_main_path = str(main_path)
        metric_path = main_path / nuboard_file.metric_folder
        if not metric_path.exists():
            metric_path.mkdir(parents=True, exist_ok=True)

        simulation_path = main_path / nuboard_file.simulation_folder
        if not simulation_path.exists():
            simulation_path.mkdir(parents=True, exist_ok=True)

        aggregator_metric_path = main_path / nuboard_file.aggregator_metric_folder
        if not aggregator_metric_path.exists():
            aggregator_metric_path.mkdir(parents=True, exist_ok=True)

        # Replace with new local path
        save_path = main_path / nuboard_filename
        nuboard_file.save_nuboard_file(save_path)
        logger.info("Updated nubBard main path in {} to {}".format(save_path, main_path))
        self._configuration_tab.add_nuboard_file_to_experiments(nuboard_file=s3_nuboard_file_result.nuboard_file)

    def _download_s3_file_contents(self, s3_result_file_contents: S3FileResultMessage, selected_s3_bucket: str) -> None:
        """
        Download s3 file contents.
        :param s3_result_file_contents: S3 file result contents.
        :param selected_s3_bucket: Selected s3 bucket name.
        """
        for index, (file_name, content) in enumerate(s3_result_file_contents.file_contents.items()):
            # Skip '..'
            if '..' in file_name:
                continue
            s3_path = os.path.join(selected_s3_bucket, file_name)
            if not file_name.endswith('/'):
                s3_connection_message = download_s3_file(
                    s3_path=s3_path,
                    s3_client=self._s3_client,
                    file_content=content,
                    save_path=self._nuplan_exp_root,
                )
            else:
                s3_connection_message = download_s3_path(
                    s3_path=s3_path, s3_client=self._s3_client, save_path=self._nuplan_exp_root
                )
            if s3_connection_message.success:
                text_message = f"Downloaded {file_name} ({index + 1} / {len(s3_result_file_contents.file_contents)})"
                logger.info(
                    "Downloaded {} / ({}/{})".format(file_name, index + 1, len(s3_result_file_contents.file_contents))
                )
                self._doc.add_next_tick_callback(partial(self._update_error_text_label, text=text_message))

    def _s3_download_button_on_click(self) -> None:
        """Function to call when the download button is click."""
        selected_s3_bucket = str(self.s3_bucket_name.text).strip()
        self.s3_download_button.label = 'Downloading...'
        self.s3_download_button.disabled = True
        self.s3_download_text_input.disabled = True
        if not selected_s3_bucket:
            self.s3_error_text.text = 'Please connect to a s3 bucket'
            self._doc.add_next_tick_callback(self._reset_s3_download_button)
            return

        selected_s3_prefix = str(self.s3_download_text_input.value).strip()
        if not selected_s3_prefix:
            self.s3_error_text.text = 'Please input a prefix'
            self._doc.add_next_tick_callback(self._reset_s3_download_button)
            return

        self._doc.add_next_tick_callback(self._s3_download_prefixes)
