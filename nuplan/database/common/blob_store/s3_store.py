from __future__ import annotations

import io
import logging
import ssl
import time
from typing import Any, BinaryIO, Optional, Tuple, Type
from urllib import parse

import botocore
import urllib3
from botocore.exceptions import BotoCoreError, NoCredentialsError
from tqdm import tqdm

from nuplan.common.utils.s3_utils import get_s3_client
from nuplan.database.common.blob_store.blob_store import BlobStore, BlobStoreKeyNotFound

logger = logging.getLogger(__name__)


class S3Store(BlobStore):
    """
    S3 blob store. Load blobs from AWS S3.
    """

    def __init__(self, s3_prefix: str, profile_name: Optional[str] = None, show_progress: bool = True) -> None:
        """
        Initialize S3Store.
        :param s3_prefix: S3 path
        :param profile_name: Profile name.
        :param show_progress: Whether to show download progress.
        """
        assert s3_prefix.startswith('s3://')
        self._s3_prefix = s3_prefix
        if not self._s3_prefix.endswith('/'):
            self._s3_prefix += '/'
        self._profile_name = profile_name

        url = parse.urlparse(self._s3_prefix)
        self._bucket = url.netloc
        self._prefix = url.path.lstrip('/')

        self._client = get_s3_client(self._profile_name)

        self._show_progress = show_progress

    def __reduce__(self) -> Tuple[Type[S3Store], Tuple[Any, ...]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class.
        """
        return self.__class__, (self._s3_prefix, self._profile_name)

    def _get_s3_location(self, key: str) -> Tuple[str, str, str]:
        """
        Get s3 location information.
        :param key: Full S3 path or bucket key of blob.
        :return: Full S3 path, bucket and key.
        """
        # Use S3 path as is if the key starts with s3:// or use default bucket/prefix to compose it
        s3_path = key if key.startswith('s3://') else f's3://{self._bucket}/{self._prefix}{key}'

        # Decompose path to bucket/key
        url = parse.urlparse(s3_path)
        bucket = url.netloc
        parsed_key = url.path.lstrip('/')

        return s3_path, bucket, parsed_key

    def get(self, key: str, check_for_compressed: bool = False) -> BinaryIO:
        """
        Get blob content.
        :param key: Full S3 path or bucket key of blob.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        :return: A file-like object, use read() to get raw bytes.
        """
        path, _, _ = self._get_s3_location(key)
        gzip_path = path + '.gzip'
        if check_for_compressed and self.exists(gzip_path):
            gzip_stream = self._get(key=gzip_path)
            content: BinaryIO = self._extract_gzip_content(gzip_stream)
        else:
            content = self._get(key=key)

        return content

    def _get(self, key: str, num_tries: int = 7) -> BinaryIO:
        """
        Get blob content from path/key.

        Note: Occasionally S3 give a ConnectionResetError or http.client.IncompleteRead
              exception. urllib3 wraps both of these in a ProtocolError. Sometimes S3 also
              gives an "ssl.SSLError: [SSL: WRONG_VERSION_NUMBER]" error. Unfortunately the
              boto3 retrying ("max_attempts") gives up when it sees any of these exceptions,
              and we have to handle retrying them ourselves. Starting with version 1.26.0,
              urllib3 wraps the ssl.SSLError into a urllib3.exceptions.SSLError.

        Note: Pytorch uses an ExceptionWrapper class that tries to "reconstruct" its wrapped
              exception, but if a new exception gets thrown *while calling the constructor* of
              the wrapped exception's type, then that new exception is raised instead of an
              instance of the wrapped exception's type. Long story short, this means some
              retryable AWS exceptions get turned into KeyErrors, so we have to catch KeyError too.

        :param key: Full S3 path or bucket key of blob.
        :param num_tries: Number of download tries.
        :return: Blob binary stream.
        """
        s3_path, bucket, key = self._get_s3_location(key)
        disable_progress = not self._show_progress

        for try_number in range(0, num_tries):
            try:
                total_length = int(self._client.head_object(Bucket=bucket, Key=key).get('ContentLength', 0))
                bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                with tqdm(
                    total=total_length,
                    desc=f'Downloading {s3_path}...',
                    bar_format=bar_format,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    disable=disable_progress,
                ) as pbar:
                    stream: BinaryIO = io.BytesIO()
                    self._client.download_fileobj(bucket, key, stream, Callback=pbar.update)
                    stream.seek(0)
                break
            except (
                urllib3.exceptions.ProtocolError,
                ssl.SSLError,
                urllib3.exceptions.SSLError,
                KeyError,
                BotoCoreError,
                # This can be a transient issue when using IAM auth in a multi-threaded environment
                NoCredentialsError,
            ) as e:
                if isinstance(e, KeyError):
                    logger.warning(f"Caught KeyError: {e}. Retrying S3 read.")

                was_last_try = try_number == (num_tries - 1)
                if was_last_try:
                    raise e
                else:
                    logger.debug(f"Retrying S3 fetch due to exception {e}")
                    time.sleep(2**try_number)
            except botocore.exceptions.ClientError as error:
                if error.response['Error']['Code'] == 'NoSuchKey':
                    message = f'{str(error)}\nS3 path not found: {s3_path}'
                    raise BlobStoreKeyNotFound(message)
                else:
                    raise RuntimeError(f"{error} Key: {s3_path}")

        return stream

    async def get_async(self, key: str) -> BinaryIO:
        """Inherited, see superclass."""
        raise NotImplementedError('Not today.')

    def save_to_disk(self, key: str, check_for_compressed: bool = False) -> None:
        """Inherited, see superclass."""
        super().save_to_disk(key, check_for_compressed=check_for_compressed)

    def exists(self, key: str) -> bool:
        """
        Tell if the blob exists.
        :param key: blob path or token.
        :return: True if the blob exists else False.
        """
        _, bucket, key = self._get_s3_location(key)

        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['ResponseMetadata']['HTTPStatusCode'] == 404:
                return False
            raise
        except BotoCoreError as e:
            logger.debug(e)
            return False

    def put(self, key: str, value: BinaryIO, ignore_if_client_error: bool = False) -> bool:
        """
        Writes content to the blobstore.
        :param key: Blob path or token.
        :param value: Data to save.
        :param ignore_if_client_error: Set to true if we want to ignore botocore client error
        """
        _, bucket, key = self._get_s3_location(key)
        successfully_stored_object = False
        try:
            response = self._client.put_object(Body=value, Bucket=bucket, Key=key)
            successfully_stored_object = response is not None
            if not successfully_stored_object:
                raise RuntimeError(f"Failed to store object to blobstore. Key : {key}")
        except botocore.exceptions.ClientError as error:  # TODO: Look into what is causing this error
            logger.info(f'{error}')
            if not ignore_if_client_error:
                raise RuntimeError(f"{error} Key: {key}")

        return successfully_stored_object
