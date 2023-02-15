import os
import pathlib
from typing import List, Optional, Tuple

import boto3


def _create_directories(local_path_name: str, directories: List[str]) -> None:
    """
    Creates directories from a list of directory names and a base path
    :param local_path_name: the base path
    :param directories: The name of the directories to create
    """
    local_path = pathlib.Path(local_path_name)
    for _dir in directories:
        (local_path / _dir).mkdir(exist_ok=True, parents=True)


def filter_paths(paths: List[str], filters: Optional[List[str]]) -> List[str]:
    """Filters a list of paths according to a list of filters.
    :param paths: The input paths.
    :param filters: The filters of the elements to keep.
    :return: The subset of paths that contain at least one of the keywords defined in filters.
    """
    return [path for path in paths if not filters or any(_filter in path for _filter in filters)]


def list_objects(bucket: str, client: boto3.client, prefix: str) -> Tuple[List[str], List[str]]:
    """
    Returns files and directories in the bucket at the given prefix.
    :param bucket: The s3 bucket
    :param client: The s3 client
    :param prefix: Prefix used to filer targets to download
    :return: A list of directories and a list of files on the bucket matching the prefix.
    """
    keys: List[str] = []
    directories: List[str] = []
    next_token = 'InitialToken'
    list_request = {'Bucket': bucket, 'Prefix': prefix}

    # List objects only lists up to 1000 files, we need to check ContinuationToken
    while next_token:
        # Request
        results = client.list_objects_v2(**list_request)

        # Extract files
        contents = results.get('Contents')
        if not contents:
            break
        for content in contents:
            target: str = content.get('Key')
            if target.endswith('/'):
                directories.append(target)
            else:
                keys.append(target)

        # Prepare next request
        next_token = results.get('NextContinuationToken')
        list_request['ContinuationToken'] = next_token

    return directories, keys


def _download_files(
    bucket: str, client: boto3.client, local_path_name: str, keys: List[str], filters: Optional[List[str]] = None
) -> None:
    """
    Downloads a list of objects from s3
    :param bucket: The s3 bucket
    :param client: The s3 client
    :param local_path_name: the base path
    :param keys: The name of the objects to download
    """
    local_path = pathlib.Path(local_path_name)
    filtered_keys = filter_paths(keys, filters)

    for key in filtered_keys:
        dest_file = local_path / key
        dest_file.parent.mkdir(exist_ok=True, parents=True)
        client.download_file(bucket, key, str(dest_file))


def s3_download_dir(
    bucket: str, client: boto3.client, prefix: str, local_path_name: str, filters: Optional[List[str]] = None
) -> None:
    """
    Downloads targets matching a prefix from s3
    :param bucket: The s3 bucket
    :param client: The s3 client
    :param prefix: Prefix used to filer targets to download
    :param local_path_name: the base path
    :param filters: Keywords to filter paths, if empty no filtering is performed.
    """
    directories, keys = list_objects(bucket, client, prefix)

    _create_directories(local_path_name, directories)
    _download_files(bucket, client, local_path_name, keys, filters)


def s3_download(prefix: str, local_path_name: str, filters: Optional[List[str]] = None) -> None:
    """
    Downloads all files matching a pattern on s3 creating a client
    :param prefix: The pattern matching prefix
    :param local_path_name: The local destination
    :param filters: Keywords to filter paths, if empty no filtering is performed.
    """
    args = {
        "region_name": "us-east-1",
    }
    if os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE") is None and os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") is None:
        args["aws_access_key_id"] = os.environ["NUPLAN_SERVER_AWS_ACCESS_KEY_ID"]
        args["aws_secret_access_key"] = os.environ["NUPLAN_SERVER_AWS_SECRET_ACCESS_KEY"]
    s3_client = boto3.client('s3', **args)
    s3_bucket = os.getenv("NUPLAN_SERVER_S3_ROOT_URL")

    assert s3_bucket, "S3 bucket not specified!"

    s3_download_dir(s3_bucket, s3_client, prefix, local_path_name, filters)
