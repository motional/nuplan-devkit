import logging
import os
from typing import List, Optional
from urllib import parse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError

logger = logging.getLogger(__name__)

DEFAULT_S3_PROFILE = os.getenv('NUPLAN_S3_PROFILE', '')


def get_s3_client(profile_name: Optional[str] = None, max_attempts: int = 10) -> boto3.client:
    """
    Start a Boto3 session and retrieve the client.
    :param profile_name: S3 profile name to use when creating the session.
    :param max_attemps: Maximum number of attempts in loading the client.
    :return: The instantiated client object.
    """
    profile_name = DEFAULT_S3_PROFILE if profile_name is None else profile_name

    try:
        session = boto3.Session(profile_name=profile_name)
    except BotoCoreError as e:
        logger.info(
            "Trying default AWS credential chain, since we got this exception "
            f"while trying to use AWS profile [{profile_name}]: {e}"
        )
        session = boto3.Session()

    config = Config(retries={"max_attempts": max_attempts})

    return session.client('s3', config=config)


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


def expand_s3_dir(s3_path: str, client: Optional[boto3.client] = None, filter_suffix: str = '') -> List[str]:
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
