import logging
import os
import ssl
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import parse

import boto3
import urllib3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ConnectTimeoutError, NoCredentialsError

# retry type stubs only support python 3.8 or earlier
from retry import retry  # type: ignore

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    urllib3.exceptions.ProtocolError,
    urllib3.exceptions.SSLError,
    ssl.SSLError,
    BotoCoreError,
    NoCredentialsError,
    ConnectTimeoutError,
)

# S3 Session is created lazily and cached globally.
# This way, it can be shared with forked processes.
S3_SESSION = None


def get_s3_session(
    profile_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> boto3.session:
    """
    Starts a Boto3 session with the specified authentication methods.
    If all three arguments are None, the default profile will be used.
    Otherwise, an attempt will be made to use the provided credentials,
      falling back to the default profile otherwise.
    :param profile_name: The profile to use, if using credentialed authentication.
    :param aws_access_key_id: The access key id to use, if using credentialed authentication.
    :param aws_secret_access_key: The secret access key to use, if using credentialed authentication.
    :return: The boto3 session.
    """
    global S3_SESSION
    # If session has already been initialized, use it.
    if S3_SESSION is not None:
        return S3_SESSION

    # If an AWS_WEB_IDENTITY_TOKEN_FILE is present, use that for credentials.
    # This file is typically present on S3 infra (e.g. EKS clusters).
    if os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE") is not None:
        logger.debug("Using AWS_WEB_IDENTITY_TOKEN_FILE for credentials.")
        S3_SESSION = boto3.Session()
        return S3_SESSION

    # If no credential information is provided, use the default authentication chain.
    # This would be typically used with AWS SSO.
    if profile_name is None and aws_access_key_id is None and aws_secret_access_key is None:
        logger.debug("Using default credentials for AWS session.")
        S3_SESSION = boto3.Session()
        return S3_SESSION

    # If profile / key information is provided, then attempt to use that for the credentials.
    # This could be the case e.g. for external users during NuPlan competition.
    logger.debug("Attempting to use credentialed authentication for S3 client...")
    args: Dict[str, Any] = {
        "profile_name": os.getenv("NUPLAN_S3_PROFILE", "") if profile_name is None else profile_name
    }

    # External users could (theoretically) have a public bucket.
    if aws_access_key_id and aws_secret_access_key:
        args['aws_access_key_id'] = aws_access_key_id
        args['aws_secret_access_key'] = aws_secret_access_key
    try:
        # If credentialed authentication is needed, a profile must be specified.
        # This can be done either in the web_identity_token_file or provided as an argument.
        # For backward compatibility, set the profile name from the environment variable if not specified.
        S3_SESSION = boto3.Session(**args)
        return S3_SESSION
    except BotoCoreError as e:
        logger.info(
            "Trying default AWS credential chain, since we got this exception "
            f"while trying to use AWS profile [{args['profile_name']}]: {e}"
        )
        S3_SESSION = boto3.Session()
        return S3_SESSION


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
    session = get_s3_session(profile_name, aws_access_key_id, aws_secret_access_key)

    config = Config(retries={"max_attempts": max_attempts})
    client = session.client("s3", config=config)

    return client


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

    client = get_s3_client()

    url = parse.urlparse(s3_path)
    response = client.list_objects(Bucket=url.netloc, Prefix=url.path.lstrip("/"))

    return "Contents" in response


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
    client = get_s3_client() if client is None else client

    url = parse.urlparse(s3_path)

    paginator = client.get_paginator("list_objects_v2")

    page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=url.path.lstrip("/"))
    filenames = [str(content["Key"]) for page in page_iterator for content in page["Contents"]]
    filenames = [f"s3://{url.netloc}/{path}" for path in filenames if path.endswith(filter_suffix)]

    return filenames


@retry(RETRYABLE_EXCEPTIONS, backoff=2, tries=7, delay=0.5, jitter=(0.5, 3))
def get_cache_metadata_paths(
    s3_path: str,
    client: Optional[boto3.client] = None,
    metadata_folder: str = "metadata",
    filter_suffix: str = "csv",
) -> List[str]:
    """
    Find metadata file paths in S3 cache path provided.
    :param s3_path: S3 path dir to expand.
    :param client: Boto3 client to use, if None create a new one.
    :param metadata_folder: Metadata folder name.
    :param filter_suffix: Optional suffix to filter S3 filenames with.
    :return: List of S3 filenames discovered.
    """
    client = get_s3_client() if client is None else client

    url = parse.urlparse(s3_path)

    paginator = client.get_paginator("list_objects_v2")

    logger.info("Attempting to find directory metadata for faster listing...")
    page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=os.path.join(url.path.lstrip("/"), metadata_folder))

    filenames = []
    try:
        filenames = [str(content["Key"]) for page in page_iterator for content in page["Contents"]]
        filenames = [f"s3://{url.netloc}/{path}" for path in filenames if path.endswith(filter_suffix)]
    except KeyError as err:
        logger.info(
            f"Error: {err}. No metadata found in directory provided! Please ensure cache contains metadata and directory provided is correct."
        )

    return filenames


def is_s3_path(candidate: Path) -> bool:
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
    if not is_s3_path(s3_path):
        raise ValueError(f"{str(s3_path)} is not an s3 path.")

    chunks = [v.strip() for v in str(s3_path).split("/") if len(v.strip()) > 0]

    bucket = chunks[1]
    path = Path("/".join(chunks[2:]))

    return bucket, path
