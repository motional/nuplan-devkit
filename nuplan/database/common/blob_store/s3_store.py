from __future__ import annotations

import io
import logging
import ssl
import time
from typing import BinaryIO, Tuple, Type
from urllib import parse

import boto3
import botocore
import urllib3
from botocore.config import Config
from botocore.exceptions import BotoCoreError
from nuplan.database.common.blob_store.blob_store import BlobStore, BlobStoreKeyNotFound

logger = logging.getLogger(__name__)


class S3Store(BlobStore):
    """
    S3 blob store. Load blobs from AWS S3.
    """

    def __init__(self, s3_prefix: str, profile_name: str = 'ml') -> None:
        """
        Initialize S3Store.
        :param s3_prefix: S3 path
        :param profile_name: Profile name.
        """
        assert s3_prefix.startswith('s3://')
        self._s3_prefix = s3_prefix
        if not self._s3_prefix.endswith('/'):
            self._s3_prefix += '/'
        self._profile_name = profile_name

        url = parse.urlparse(self._s3_prefix)
        self._bucket = url.netloc
        self._prefix = url.path.lstrip('/')

        # This `try-except` looks for AWS credentials in this order,  using the first
        # credentials it finds:
        # 1. `profile_name` profile (likely "ml")
        # 2. AWS environment variables
        # 3. [default] profile
        # 4. EC2 instance role
        # They will only use that EC2 instance role if they fail to find any credentials
        # at the [ml] or [default] profiles in ~/.aws/credentials, the AWS_ACCESS_KEY_ID
        # and AWS_SECRET_ACCESS_KEY environment variables.
        try:
            self._session = boto3.Session(profile_name=profile_name)
        except BotoCoreError as e:
            # It's most likely that we caught a ProfileNotFound or NoCredentialsError,
            # but we catch any BotoCoreError since we we might as well retry no matter
            # what the exception was.
            logger.info(f"Trying default AWS credential chain, since we got this exception "
                        f"while trying to use AWS profile [{profile_name}]: {e}")
            # This is poorly documented in boto3, but if you look at
            # `botocore.credentials.create_credential_resolver` you can see all the
            # places this looks for credentials.
            # Note: this will not throw an exception until we actually try to *use* the
            # session, even if it can't find credentials or found bad credentials.
            self._session = boto3.Session()
        config = Config(retries={"max_attempts": 10})
        self._client = self._session.client('s3', config=config)

    def __reduce__(self) -> Tuple[Type[S3Store], Tuple[str, str]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class.
        """
        return self.__class__, (self._s3_prefix, self._profile_name)

    def get(self, key: str, check_for_compressed: bool = False) -> BinaryIO:
        """
        Get blob content.
        :param key: Blob path or token.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        :return: A file-like object, use read() to get raw bytes.
        """
        path = self._s3_prefix + key
        gzip_path = path + '.gzip'
        if check_for_compressed and self.exists(gzip_path):
            gzip_stream = self._get(gzip_path)
            content: BinaryIO = self._extract_gzip_content(gzip_stream)
        else:
            content = self._get(key)

        return content

    def _get(self, path: str) -> BinaryIO:
        """
        Get blob content.
        :param path: File path to download.
        :return: Blob binary stream.
        """
        # Occasionally S3 gives us a ConnectionResetError or http.client.IncompleteRead
        # exception. urllib3 wraps both of these in a ProtocolError. Sometimes S3 also
        # gives an "ssl.SSLError: [SSL: WRONG_VERSION_NUMBER]" error. Unfortunately the
        # boto3 retrying ("max_attempts") gives up when it sees any of these exceptions,
        # and we have to handle retrying them ourselves.
        #
        # Starting with version 1.26.0, urllib3 wraps the ssl.SSLError into a
        # urllib3.exceptions.SSLError.
        path = self._prefix + path
        t0 = time.time()
        num_tries = 7
        stream: BinaryIO = None  # type: ignore
        for try_number in range(0, num_tries):
            try:
                response = self._client.get_object(Bucket=self._bucket, Key=path)
                stream = io.BytesIO(response['Body'].read())
                break
            # Pytorch uses an ExceptionWrapper class that tries to "reconstruct" its wrapped
            # exception, but if a new exception gets thrown *while calling the constructor* of
            # the wrapped exception's type, then that new exception is raised instead of an
            # instance of the wrapped exception's type. Long story short, this means some
            # retryable AWS exceptions get turned into KeyErrors, so we have to catch KeyError too.
            except (urllib3.exceptions.ProtocolError, ssl.SSLError,
                    urllib3.exceptions.SSLError, KeyError, BotoCoreError) as e:
                if isinstance(e, KeyError):
                    logger.warning(f"Caught KeyError: {e}. Retrying S3 read.")

                was_last_try = try_number == (num_tries - 1)
                if was_last_try:
                    raise e
                else:
                    logger.debug(f"Retrying S3 fetch due to exception {e}")
                    time.sleep(2 ** try_number)
            except botocore.exceptions.ClientError as error:
                if error.response['Error']['Code'] == 'NoSuchKey':
                    raise BlobStoreKeyNotFound(error)
                else:
                    raise RuntimeError(f"{error} Key: {path}.")

        logger.debug("Done fetching {} in {} seconds.".format(path, time.time() - t0))
        return stream

    async def get_async(self, key: str) -> BinaryIO:
        """ Inherited, see superclass. """
        raise NotImplementedError('Not today.')

    def save_to_disk(self, key: str, check_for_compressed: bool = False) -> None:
        """ Inherited, see superclass. """
        super().save_to_disk(key, check_for_compressed=check_for_compressed)

    def exists(self, key: str) -> bool:
        """
        Tell if the blob exists.
        :param key: blob path or token.
        :return: True if the blob exists else False.
        """
        path = self._prefix + key
        try:
            self._client.head_object(Bucket=self._bucket, Key=path)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['ResponseMetadata']['HTTPStatusCode'] == 404:
                return False
            raise
        except BotoCoreError as e:
            logger.debug(e)
            return False

    def put(self, key: str, value: BinaryIO) -> None:
        """
        Writes content to the blobstore.
        :param key: Blob path or token.
        :param value: Data to save.
        """
        path = self._s3_prefix + key
        self._client.put_object(Body=value, Bucket=self._bucket, Key=path)
