import asyncio
import logging
import os
import ssl
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib import parse

import aioboto3
import boto3
import urllib3
from botocore.config import Config
from botocore.errorfactory import ClientError
from botocore.exceptions import BotoCoreError, ConnectTimeoutError, NoCredentialsError

# retry type stubs only support python 3.8 or earlier
from retry import retry  # type: ignore

logger = logging.getLogger(__name__)

# S3 sessions are created lazily and cached globally.
# This way, it can be shared with forked processes.
G_ASYNC_SESSION = None
G_SYNC_SESSION = None

RETRYABLE_EXCEPTIONS = (
    urllib3.exceptions.ProtocolError,
    urllib3.exceptions.SSLError,
    ssl.SSLError,
    BotoCoreError,
    NoCredentialsError,
    ConnectTimeoutError,
)


def _get_session_internal(
    profile_name: Optional[str],
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    create_session_func: Callable[..., Union[boto3.Session, aioboto3.Session]],
    set_session_func: Callable[[Union[boto3.Session, aioboto3.Session]], None],
) -> Union[boto3.Session, aioboto3.Session]:
    """
    Get synchronous boto3 session.
    :param profile_name: Optional profile name to authenticate with.
    :param aws_access_key_id: Optional access key to authenticate with.
    :param aws_secret_access_key: Optional secret access key to authenticate with.
    :param create_session_func: Session creation function.
    :param set_session_func: Session caching function.
    :return: Session object.
    """
    args: Dict[str, Any] = {}

    # If an AWS_WEB_IDENTITY_TOKEN_FILE is present, use that for credentials.
    # This file is typically present on S3 infra (e.g. EKS clusters).
    if os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE") is not None:
        logger.debug("Using AWS_WEB_IDENTITY_TOKEN_FILE for credentials.")

    # If no credential information is provided, use the default authentication chain.
    # This would be typically used with AWS SSO.
    elif profile_name is None and aws_access_key_id is None and aws_secret_access_key is None:
        logger.debug("Using default credentials for AWS session.")

    # If profile / key information is provided, then attempt to use that for the credentials.
    # This could be the case e.g. for external users during NuPlan competition.
    else:
        logger.debug("Attempting to use credentialed authentication for S3 client...")
        args = {"profile_name": os.getenv("NUPLAN_S3_PROFILE", "") if profile_name is None else profile_name}

        # External users could (theoretically) have a public bucket.
        if aws_access_key_id and aws_secret_access_key:
            args['aws_access_key_id'] = aws_access_key_id
            args['aws_secret_access_key'] = aws_secret_access_key

    try:
        # If credentialed authentication is needed, a profile must be specified.
        # This can be done either in the web_identity_token_file or provided as an argument.
        # For backward compatibility, set the profile name from the environment variable if not specified.
        session = create_session_func(**args)
        set_session_func(session)
    except BotoCoreError as e:
        # If we tried authenticating with a profile, try again with default credentials chain.
        if "profile_name" in args:
            logger.info(
                "Trying default AWS credential chain, since we got this exception "
                f"while trying to use AWS profile [{args['profile_name']}]: {e}"
            )

        session = create_session_func()
        set_session_func(session)

    return session


def get_async_s3_session(
    profile_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    force_new: bool = False,
) -> aioboto3.Session:
    """
    Get synchronous boto3 session.
    :param profile_name: Optional profile name to authenticate with.
    :param aws_access_key_id: Optional access key to authenticate with.
    :param aws_secret_access_key: Optional secret access key to authenticate with.
    :param force_new: If true, ignore any cached  session and get a new one.
                      Any existing cached session will be overwritten.
    :return: Session object.
    """
    global G_ASYNC_SESSION
    if not force_new and G_ASYNC_SESSION is not None:
        return G_ASYNC_SESSION

    def _set_async_session_func(session: aioboto3.Session) -> None:
        global G_ASYNC_SESSION
        G_ASYNC_SESSION = session

    def _create_session_func(**kwargs: Any) -> aioboto3.Session:
        return aioboto3.Session(**kwargs)

    return _get_session_internal(
        profile_name, aws_access_key_id, aws_secret_access_key, _create_session_func, _set_async_session_func
    )


def _get_sync_session(
    profile_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    force_new: bool = False,
) -> boto3.Session:
    """
    Get synchronous boto3 session.
    :param profile_name: Optional profile name to authenticate with.
    :param aws_access_key_id: Optional access key to authenticate with.
    :param aws_secret_access_key: Optional secret access key to authenticate with.
    :param force_new: If true, ignore any cached session and get a new one.
                      Any existing cached session will be overwritten.
    :return: Session object.
    """
    global G_SYNC_SESSION
    if not force_new and G_SYNC_SESSION is not None:
        return G_SYNC_SESSION

    def _set_sync_session_func(session: boto3.Session) -> None:
        global G_SYNC_SESSION
        G_SYNC_SESSION = session

    def _create_session_func(**kwargs: Any) -> aioboto3.Session:
        return boto3.Session(**kwargs)

    return _get_session_internal(
        profile_name, aws_access_key_id, aws_secret_access_key, _create_session_func, _set_sync_session_func
    )


# No need to retry - only retryable operation would be the IAM cache,
#  which is already covered by retry decorators.
def get_s3_client(
    profile_name: Optional[str] = None,
    max_attempts: int = 10,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> boto3.client:
    """
    Start a Boto3 session and retrieve the client.
    :param profile_name: S3 profile name to use when creating the session.
    :param aws_access_key_id: Aws access key id.
    :param aws_secret_access_key: Aws secret access key.
    :param max_attempts: Maximum number of attempts in loading the client.
    :return: The instantiated client object.
    """
    session = _get_sync_session(profile_name, aws_access_key_id, aws_secret_access_key)

    config = Config(retries={"max_attempts": max_attempts})
    client = session.client("s3", config=config)

    return client


def _trim_leading_slash_if_exists(path: Union[str, Path]) -> Path:
    """
    Trims the leading slash in a path if it exists.
    :param path: The path to trim.
    :return: The trimmed path.
    """
    path_str = str(path)
    if path_str == "/":
        raise ValueError("Path is the root path '/'. This should never happen.")
    path_str = path_str[1:] if path_str.startswith("/") else path_str
    return Path(path_str)


def is_s3_path(candidate: Union[Path, str]) -> bool:
    """
    Returns true if the path points to a location in S3, false otherwise.
    :param candidate: The candidate path.
    :return: True if the path points to a location in S3, false otherwise.
    """
    candidate_str = str(candidate)
    return candidate_str.startswith("s3:/")


def split_s3_path(s3_path: Path) -> Tuple[str, Path]:
    """
    Splits a S3 path into a (bucket, path) set of identifiers.
    :param s3_path: The full S3 path.
    :return: A tuple of (bucket, path).
    """
    # Expect path of the form:
    # s3://ml-caches/folder/folder2/folder3/file.txt
    #
    # Would result in:
    #  bucket = "ml-caches"
    #  path = "folder/folder2/folder3/file.txt"
    if not is_s3_path(s3_path):
        raise ValueError(f"{str(s3_path)} is not an s3 path.")

    chunks = [v.strip() for v in str(s3_path).split("/") if len(v.strip()) > 0]

    bucket = chunks[1]
    path = Path("/".join(chunks[2:]))

    return bucket, path


def download_directory_from_s3(local_dir: Path, s3_key: Path, s3_bucket: str) -> None:
    """
    Downloads a directory to the local machine.
    :param local_dir: The directory to which to download.
    :param s3_key: The directory in S3 to download, without the bucket.
    :param s3_bucket: The bucket name to use.
    """
    asyncio.run(download_directory_from_s3_async(local_dir, s3_key, s3_bucket))


async def download_directory_from_s3_async(local_dir: Path, s3_key: Path, s3_bucket: str) -> None:
    """
    Downloads a directory to the local machine asynchronously.
    :param local_dir: The directory to download.
    :param s3_key: The directory in S3 to download, without the bucket.
    :param s3_bucket: The bucket name to use.
    """
    paths = await list_files_in_s3_directory_async(s3_key, s3_bucket)
    tasks: List[asyncio.Task[None]] = []
    for path in paths:
        local_path = local_dir / _trim_leading_slash_if_exists(Path(str(path).replace(str(s3_key), "")))
        tasks.append(asyncio.create_task(download_file_from_s3_async(local_path, path, s3_bucket)))

    _ = await asyncio.gather(*tasks)


def download_file_from_s3(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
    """
    Downloads a file to local disk from S3.
    :param local_path: The path to which to download.
    :param s3_key: The S3 path from which to download, without the bucket.
    :param s3_bucket: The bucket name to use.
    """
    asyncio.run(download_file_from_s3_async(local_path, s3_key, s3_bucket))


@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
async def download_file_from_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
    """
    Downloads a file to local disk from S3 asynchronously.
    :param local_path: The path to which to download.
    :param s3_key: The S3 path from which to download, without the bucket.
    :param s3_bucket: The bucket name to use.
    """
    if not local_path.parent.exists():
        local_path.parent.mkdir(exist_ok=True, parents=True)

    session = get_async_s3_session()
    async with session.client("s3") as async_s3_client:
        logger.info(f"Downloading {s3_key} to {local_path} in bucket {s3_bucket}...")
        await async_s3_client.download_file(s3_bucket, str(s3_key), str(local_path))


@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
def upload_file_to_s3(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
    """
    Uploads a file from the local disk to S3.
    :param local_path: The local path to the file.
    :param s3_key: The S3 path for the file, without the bucket.
    :param s3_bucket: The name of the bucket to write to.
    """
    asyncio.run(upload_file_to_s3_async(local_path, s3_key, s3_bucket))


@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
async def upload_file_to_s3_async(local_path: Path, s3_key: Path, s3_bucket: str) -> None:
    """
    Uploads a file from local disk to S3 asynchronously.
    :param local_path: The local path to the file.
    :param s3_key: The S3 path for the file, without the bucket.
    :param s3_bucket: The name of the bucket to write to.
    """
    session = get_async_s3_session()
    async with session.client("s3") as async_s3_client:
        logger.info(f"Uploading {local_path} to {s3_key} in bucket {s3_bucket}...")
        await async_s3_client.upload_file(str(local_path), s3_bucket, str(s3_key))


@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
def delete_file_from_s3(s3_key: Path, s3_bucket: str) -> None:
    """
    Deletes a single file from S3.
    :param s3_key: The path pointing to the file, without the bucket.
    :param s3_bucket: The name of the bucket.
    """
    asyncio.run(delete_file_from_s3_async(s3_key, s3_bucket))


@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
async def delete_file_from_s3_async(s3_key: Path, s3_bucket: str) -> None:
    """
    Deletes a single file from S3 asynchronously.
    :param s3_key: The path pointing to the file, without the bucket.
    :param s3_bucket: The name of the bucket.
    """
    session = get_async_s3_session()
    async with session.client("s3") as async_s3_client:
        await async_s3_client.delete_object(Bucket=s3_bucket, Key=str(s3_key))


def read_text_file_contents_from_s3(s3_key: Path, s3_bucket: str) -> str:
    """
    Reads the entire contents of a text file from S3.
    :param s3_key: The path pointing to the file, without the bucket.
    :param s3_bucket: The name of the bucket.
    :return: The contents of the file, decoded as a UTF-8 string.
    """
    result: str = asyncio.run(read_text_file_contents_from_s3_async(s3_key, s3_bucket))
    return result


async def read_text_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> str:
    """
    Reads the entire contents of a text file from S3 asynchronously.
    :param s3_key: The path pointing to the file, without the bucket.
    :param s3_bucket: The name of the bucket.
    :return: The contents of the file, decoded as a UTF-8 string.
    """
    result_binary: bytes = await read_binary_file_contents_from_s3_async(s3_key, s3_bucket)
    return result_binary.decode("utf-8")


def read_binary_file_contents_from_s3(s3_key: Path, s3_bucket: str) -> bytes:
    """
    Reads the entire contents of a file from S3.
    :param s3_key: The path pointing to the file, without the bucket.
    :param s3_bucket: The name of the bucket.
    :return: The contents of the file.
    """
    result: bytes = asyncio.run(read_binary_file_contents_from_s3_async(s3_key, s3_bucket))
    return result


@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
async def read_binary_file_contents_from_s3_async(s3_key: Path, s3_bucket: str) -> bytes:
    """
    Reads the entire contents of a file from S3 asynchronously.
    :param s3_key: The path pointing to the file, without the bucket.
    :param s3_bucket: The name of the bucket.
    :return: The contents of the file.
    """
    with tempfile.NamedTemporaryFile() as fp:
        file_name = fp.name

        session = get_async_s3_session()
        async with session.client("s3") as async_s3_client:
            await async_s3_client.download_file(s3_bucket, str(s3_key), file_name)

            with open(file_name, "rb") as second_fp:
                contents = second_fp.read()

            return contents


@retry(RETRYABLE_EXCEPTIONS, backoff=2, tries=7, delay=0.5, jitter=(0.5, 3))
def check_s3_path_exists(s3_path: Optional[str]) -> bool:
    """
    Check whether the S3 path exists.
    If "None" is passed, then the return will be false, because a "None" path will never exist.
    :param s3_path: S3 path to check.
    :return: Whether the path exists or not.
    """
    # We allow None as an argument because many times this is used in conjunction with Hydra.
    # Something like
    #  assert(check_s3_path_exists(cfg.some_path))
    #
    # This can be None if not specified in config.
    # It would be a common footgun if the user had to check for not None.
    if s3_path is None:
        return False

    result: bool = asyncio.run(check_s3_path_exists_async(s3_path))
    return result


async def check_s3_path_exists_async(s3_path: str) -> bool:
    """
    Check whether the S3 path exists.
    :param s3_path: S3 path to check.
    :return: Whether the path exists or not.
    """
    session = get_async_s3_session()
    async with session.client("s3") as async_s3_client:
        url = parse.urlparse(s3_path)
        response = await async_s3_client.list_objects(Bucket=url.netloc, Prefix=url.path.lstrip("/"))
        return "Contents" in response


def check_s3_object_exists(s3_key: Path, s3_bucket: str) -> bool:
    """
    Checks if an object in S3 exists.
    Returns False if the path is to a directory.
    :param s3_key: The path to list, without the bucket.
    :param s3_bucket: The bucket to list.
    :return: True if the object exists, false otherwise.
    """
    result: bool = asyncio.run(check_s3_object_exists_async(s3_key, s3_bucket))
    return result


@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
async def check_s3_object_exists_async(s3_key: Path, s3_bucket: str) -> bool:
    """
    Checks if an object in S3 exists asynchronously.
    Returns False if the path is to a directory.
    :param s3_key: The path to list, without the bucket.
    :param s3_bucket: The bucket to list.
    :return: True if the object exists, false otherwise.
    """
    session = get_async_s3_session()
    async with session.client("s3") as async_s3_client:
        try:
            await async_s3_client.head_object(Bucket=s3_bucket, Key=str(s3_key))
            return True
        except ClientError:
            return False


@retry(RETRYABLE_EXCEPTIONS, backoff=2, tries=7, delay=0.5, jitter=(0.5, 3))
def expand_s3_dir(
    s3_path: str,
    client: Optional[boto3.client] = None,
    filter_suffix: str = "",
) -> List[str]:
    """
    Expand S3 path dir to a list of S3 path files.
    :param s3_path: S3 path dir to expand.
    :param client: Boto3 client to use, if None create a new one.
    :param filter_suffix: Optional suffix to filter S3 filenames with.
    :return: List of S3 filenames discovered.
    """
    logger.warning("Function expand_s3_dir will soon be removed in favor of list_files_in_s3_directory")

    client = get_s3_client() if client is None else client

    url = parse.urlparse(s3_path)

    paginator = client.get_paginator("list_objects_v2")

    page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=url.path.lstrip("/"))
    filenames = [str(content["Key"]) for page in page_iterator for content in page["Contents"]]
    filenames = [f"s3://{url.netloc}/{path}" for path in filenames if path.endswith(filter_suffix)]

    return filenames


def list_files_in_s3_directory(s3_key: Path, s3_bucket: str, filter_suffix: str = "") -> List[Path]:
    """
    Lists the files available in a particular S3 directory.
    :param s3_key: The path to list, without the bucket.
    :param s3_bucket: The bucket to list.
    :param filter_suffix: Optional suffix to filter S3 filenames with.
    :return: The s3 keys of files in the folder.
    """
    result: List[Path] = asyncio.run(list_files_in_s3_directory_async(s3_key, s3_bucket, filter_suffix))
    return result


@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
async def list_files_in_s3_directory_async(s3_key: Path, s3_bucket: str, filter_suffix: str = "") -> List[Path]:
    """
    Lists the files available in a particular S3 directory asynchronously.
    :param s3_key: The path to list, without the bucket.
    :param s3_bucket: The bucket to list.
    :param filter_suffix: Optional suffix to filter S3 filenames with.
    :return: The s3 keys of files in the folder.
    """
    session = get_async_s3_session()
    async with session.client("s3") as async_s3_client:
        paginator = async_s3_client.get_paginator("list_objects_v2")

        page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=str(s3_key))
        filepaths = []
        async for page in page_iterator:
            if "Contents" not in page:
                continue

            for content in page["Contents"]:
                filename = str(content["Key"])

                if filename.endswith(filter_suffix):
                    filepaths.append(Path(filename))

        return filepaths


def get_cache_metadata_paths(
    s3_key: Path,
    s3_bucket: str,
    metadata_folder: str = "metadata",
    filter_suffix: str = "csv",
) -> List[str]:
    """
    Find metadata file paths in S3 cache path provided.
    :param s3_key: The path of cache outputs.
    :param s3_bucket: The bucket of cache outputs.
    :param metadata_folder: Metadata folder name.
    :param filter_suffix: Optional suffix to filter S3 filenames with.
    :return: List of S3 filenames discovered.
    """
    result: List[str] = asyncio.run(get_cache_metadata_paths_async(s3_key, s3_bucket, metadata_folder, filter_suffix))
    return result


@retry(RETRYABLE_EXCEPTIONS, backoff=2, tries=7, delay=0.5, jitter=(0.5, 3))
async def get_cache_metadata_paths_async(
    s3_key: Path,
    s3_bucket: str,
    metadata_folder: str = "metadata",
    filter_suffix: str = "csv",
) -> List[str]:
    """
    Find metadata file paths in S3 cache path provided asynchronously.
    :param s3_key: The path of cache outputs.
    :param s3_bucket: The bucket of cache outputs.
    :param metadata_folder: Metadata folder name.
    :param filter_suffix: Optional suffix to filter S3 filenames with.
    :return: List of S3 filenames discovered.
    """
    filepaths = []
    try:
        filepaths = await list_files_in_s3_directory_async(s3_key / metadata_folder, s3_bucket, filter_suffix)
        s3_paths = [f"s3://{s3_bucket}/{str(path)}" for path in filepaths]
    except KeyError as err:
        logger.info(
            "Error: %s. No metadata found in directory provided! Please ensure cache contains metadata and directory provided is correct.",
            err,
        )

    return s3_paths
