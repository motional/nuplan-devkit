from __future__ import annotations

import io
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import parse

import boto3
import numpy as np
from boto3.exceptions import Boto3Error

from nuplan.common.utils.s3_utils import get_s3_client
from nuplan.planning.nuboard.base.data_class import NuBoardFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class S3FileContent:
    """S3 file contents."""

    filename: Optional[str] = None  # Full file name
    last_modified: Optional[datetime] = None  # Last modified date time
    size: Optional[int] = None  # File size

    @property
    def date_string(self) -> Optional[str]:
        """Return date string format."""
        if not self.last_modified:
            return None
        return self.last_modified.strftime("%m/%d/%Y %H:%M:%S %Z")

    @property
    def last_modified_day(self) -> Optional[str]:
        """Return last modified day."""
        if not self.last_modified:
            return None
        datetime_now = datetime.now(timezone.utc)
        difference_day = (datetime_now - self.last_modified).days
        if difference_day == 0:
            return 'Less than 24 hours'
        elif difference_day < 30:
            return f'{difference_day} days ago'
        elif 30 <= difference_day < 60:
            return 'a month ago'
        else:
            return f'{difference_day/30} months ago'

    def kb_size(self, decimals: int = 2) -> Optional[float]:
        """
        Return file size in KB.
        :param decimals: Decimal points.
        """
        if not self.size:
            return None
        return float(np.round(self.size / 1024, decimals))

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the class.
        :return A dict of object variables.
        """
        return {'filename': self.filename, 'last_modified': str(self.last_modified), 'size': self.size}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> S3FileContent:
        """
        Deserialize data to s3 file content.
        :param data: A dictionary of data.
        :return S3FileContent after loaded the data.
        """
        return S3FileContent(
            filename=data['filename'], last_modified=datetime.fromisoformat(data['last_modified']), size=data['size']
        )


@dataclass(frozen=True)
class S3ConnectionStatus:
    """Connection status for s3."""

    return_message: str
    success: bool


@dataclass(frozen=True)
class S3FileResultMessage:
    """Data class to save aws return messages and file contents."""

    s3_connection_status: S3ConnectionStatus
    file_contents: Dict[str, S3FileContent]


@dataclass(frozen=True)
class S3NuBoardFileResultMessage:
    """Data class to save aws return messages and nuboard file."""

    s3_connection_status: S3ConnectionStatus
    nuboard_filename: Optional[str] = None
    nuboard_file: Optional[NuBoardFile] = None


def check_s3_nuboard_files(
    s3_file_contents: Dict[str, S3FileContent], s3_path: str, s3_client: boto3.client
) -> S3NuBoardFileResultMessage:
    """
    Return True in the message if there is a nuboard file and can load into nuBoard.
    :param s3_file_contents: S3 prefix with a dictionary of s3 file name and their contents.
    :Param s3_path: S3 Path starts with s3://.
    :param s3_client: s3 client session.
    :return S3NuBoardFileResultMessage to indicate if there is available nuboard file in the s3 prefix.
    """
    success = False
    return_message = "No available nuboard files in the prefix"
    nuboard_file = None
    nuboard_filename = None
    if not s3_path.endswith('/'):
        s3_path = s3_path + '/'
    url = parse.urlparse(s3_path)
    for file_name, file_content in s3_file_contents.items():
        if file_name.endswith(NuBoardFile.extension()):
            try:
                nuboard_object = s3_client.get_object(Bucket=url.netloc, Key=file_name)
                file_stream = io.BytesIO(nuboard_object['Body'].read())
                nuboard_data = pickle.load(file_stream)
                nuboard_file = NuBoardFile.deserialize(nuboard_data)
                file_stream.close()
                nuboard_filename = Path(file_name).name
                return_message = f'Found available nuboard file: {nuboard_filename}'
                success = True
                break
            except Exception as e:
                logger.info(str(e))
                continue

    return S3NuBoardFileResultMessage(
        s3_connection_status=S3ConnectionStatus(success=success, return_message=return_message),
        nuboard_filename=nuboard_filename,
        nuboard_file=nuboard_file,
    )


def get_s3_file_contents(
    s3_path: str, client: Optional[boto3.client] = None, delimiter: str = '/', include_previous_folder: bool = False
) -> S3FileResultMessage:
    """
    Get folders and files contents in the provided s3 path provided.
    :param s3_path: S3 path dir to expand.
    :param client: Boto3 client to use, if None create a new one.
    :param delimiter: Delimiter for path.
    :param include_previous_folder: Set True to include '..' as previous folder.
    :return: Dict of file contents.
    """
    return_message = "Connect successfully"
    file_contents: Dict[str, S3FileContent] = {}
    try:
        client = get_s3_client() if client is None else client
        if not s3_path.endswith('/'):
            s3_path = s3_path + '/'
        url = parse.urlparse(s3_path)
        paginator = client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=url.path.lstrip('/'), Delimiter=delimiter)
        previous_folder = os.path.join(url.path.lstrip('/'), '..')
        if previous_folder != '..' and include_previous_folder:
            file_contents[previous_folder] = S3FileContent(filename=previous_folder)

        for page in page_iterator:
            # Get folders
            for obj in page.get('CommonPrefixes', []):
                file_contents[obj['Prefix']] = S3FileContent(filename=obj['Prefix'])

            # Get files, start from the second file to skip the current folder
            for content in page.get('Contents', []):
                file_name = str(content['Key'])
                if file_name == url.path.lstrip('/'):
                    continue
                file_contents[file_name] = S3FileContent(
                    filename=file_name, last_modified=content['LastModified'], size=content['Size']
                )
        success = True
    except Exception as err:
        logger.info("Error: {}".format(err))
        return_message = f"{err}"
        success = False
    s3_connection_status = S3ConnectionStatus(return_message=return_message, success=success)
    s3_file_result_message = S3FileResultMessage(s3_connection_status=s3_connection_status, file_contents=file_contents)
    return s3_file_result_message


def download_s3_file(
    s3_path: str, file_content: S3FileContent, s3_client: boto3.client, save_path: str
) -> S3ConnectionStatus:
    """
    Download a s3 file given a s3 full path.
    :param s3_path: S3 full path.
    :param file_content: File content info.
    :param s3_client: A connecting S3 client.
    :param save_path: Local save path.
    :return S3 connection status to indicate status of s3 connection.
    """
    return_message = f'Downloaded {s3_path}'
    try:
        if s3_path.endswith('/'):
            return S3ConnectionStatus(success=False, return_message=f'{s3_path} is not a file')
        url = parse.urlparse(s3_path)
        file_name = file_content.filename if file_content.filename is not None else ''
        download_file_name = Path(save_path, file_name)
        remote_file_size = file_content.size if file_content.size is not None else 0
        local_file_size = os.path.getsize(str(download_file_name)) if download_file_name.exists() else 0
        # Skip if the file exist and file size is similar
        if not (download_file_name.exists()) or local_file_size != float(remote_file_size):
            s3_client.download_file(url.netloc, file_name, str(download_file_name))
        success = True
    except Exception as e:
        raise Boto3Error(e)

    return S3ConnectionStatus(success=success, return_message=return_message)


def download_s3_path(s3_path: str, s3_client: boto3.client, save_path: str, delimiter: str = '/') -> S3ConnectionStatus:
    """
    Download a s3 path recursively given a s3 full path.
    :param s3_path: S3 full path.
    :param s3_client: A connecting S3 client.
    :param save_path: Local save path.
    :param delimiter: Delimiter to split folders.
    :return S3 connection status to indicate status of s3 connection.
    """
    return_message = f'Downloaded {s3_path}'
    try:
        if not s3_path.endswith('/'):
            s3_path = s3_path + '/'
        url = parse.urlparse(s3_path)

        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=url.path.lstrip('/'), Delimiter=delimiter)
        for page in page_iterator:
            # Get folders
            common_prefixes = page.get('CommonPrefixes', [])
            for sub_folder in common_prefixes:
                sub_s3_path = os.path.join('s3://', url.netloc, sub_folder['Prefix'])
                local_save_sub_path = Path(save_path, sub_folder['Prefix'])
                local_save_sub_path.mkdir(parents=True, exist_ok=True)
                # Recursively download folder
                download_s3_path(s3_client=s3_client, s3_path=sub_s3_path, save_path=save_path)

            # Get files
            contents = page.get('Contents', [])
            for content in contents:
                file_name = str(content['Key'])
                file_size = content['Size']
                last_modified = content['LastModified']
                s3_file_path = os.path.join('s3://', url.netloc, file_name)
                local_folder = Path(save_path, file_name)
                local_folder.parents[0].mkdir(exist_ok=True, parents=True)
                file_content = S3FileContent(filename=file_name, size=file_size, last_modified=last_modified)
                download_s3_file(
                    s3_path=s3_file_path, file_content=file_content, s3_client=s3_client, save_path=save_path
                )
        success = True
    except Exception as e:
        raise Boto3Error(e)

    s3_connection_status = S3ConnectionStatus(success=success, return_message=return_message)
    return s3_connection_status
