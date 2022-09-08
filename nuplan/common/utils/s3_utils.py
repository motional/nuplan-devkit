import logging
import os
import threading
from typing import Any, Dict, List, Optional
from urllib import parse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError

logger = logging.getLogger(__name__)

DEFAULT_S3_PROFILE = os.getenv('NUPLAN_S3_PROFILE', '')

# In some scenarios, our codebase requests multiple copies of the S3 client with the same parameters
# This can lead to large amounts of memory usage.
#
# Create a flyweight to re-use clients with matching parameters.
S3_CLIENTS: Dict[str, Any] = {}
S3_CLIENTS_LOCK_OBJ = threading.Lock()


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
    global S3_CLIENTS
    global S3_CLIENTS_LOCK_OBJ
    profile_name = DEFAULT_S3_PROFILE if profile_name is None else profile_name
    aws_credentials_configs = {}
    if aws_access_key_id and aws_secret_access_key:
        aws_credentials_configs['aws_access_key_id'] = aws_access_key_id
        aws_credentials_configs['aws_secret_access_key'] = aws_secret_access_key
    with S3_CLIENTS_LOCK_OBJ:
        if profile_name not in S3_CLIENTS:
            try:
                session = boto3.Session(profile_name=profile_name, **aws_credentials_configs)
            except BotoCoreError as e:
                logger.info(
                    "Trying default AWS credential chain, since we got this exception "
                    f"while trying to use AWS profile [{profile_name}]: {e}"
                )
                session = boto3.Session()

            config = Config(retries={"max_attempts": max_attempts})

            S3_CLIENTS[profile_name] = session.client('s3', config=config, **aws_credentials_configs)

    return S3_CLIENTS[profile_name]


def check_s3_path_exists(s3_path: str) -> bool:
    """
    Check whether the S3 path exists.
    :param s3_path: S3 path to check.
    :return: Whether the path exists or not.
    """
    client = get_s3_client()

    url = parse.urlparse(s3_path)
    response = client.list_objects(Bucket=url.netloc, Prefix=url.path.lstrip('/'))

    return 'Contents' in response


def expand_s3_dir(
    s3_path: str,
    client: Optional[boto3.client] = None,
    filter_suffix: str = '',
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

    paginator = client.get_paginator('list_objects_v2')

    page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=url.path.lstrip('/'))
    filenames = [str(content['Key']) for page in page_iterator for content in page['Contents']]
    filenames = [f's3://{url.netloc}/{path}' for path in filenames if path.endswith(filter_suffix)]

    return filenames


def get_cache_metadata_paths(
    s3_path: str,
    client: Optional[boto3.client] = None,
    metadata_folder: str = 'metadata',
    filter_suffix: str = 'csv',
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

    paginator = client.get_paginator('list_objects_v2')

    logger.info('Attempting to find directory metadata for faster listing...')
    page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=os.path.join(url.path.lstrip('/'), metadata_folder))

    filenames = []
    try:
        filenames = [str(content['Key']) for page in page_iterator for content in page['Contents']]
        filenames = [f's3://{url.netloc}/{path}' for path in filenames if path.endswith(filter_suffix)]
    except KeyError as err:
        logger.info(
            f'Error: {err}. No metadata found in directory provided! Please ensure cache contains metadata and directory provided is correct.'
        )

    return filenames
