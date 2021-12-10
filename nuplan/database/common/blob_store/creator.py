import logging
import os
from typing import NamedTuple, Optional

import requests
from nuplan.database.common.blob_store.blob_store import BlobStore
from nuplan.database.common.blob_store.cache_store import CacheStore
from nuplan.database.common.blob_store.http_store import HttpStore
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.common.blob_store.s3_store import S3Store

logger = logging.getLogger(__name__)

RemoteConfig = NamedTuple('RemoteConfig', [
    ('http_root_url', Optional[str]),
    ('s3_root_url', Optional[str]),
    ('s3_profile', Optional[str])
])


nuplandb_conf = RemoteConfig(
    http_root_url=os.getenv('NUPLAN_HTTP_ROOT_URL', ''),
    s3_root_url=os.getenv('NUPLAN_S3_ROOT_URL', ''),
    s3_profile=os.getenv('NUPLAN_S3_PROFILE', ''))

mapsdb_conf = RemoteConfig(
    http_root_url=os.getenv('NUPLAN_HTTP_ROOT_URL', ''),
    s3_root_url=os.getenv('NUPLAN_MAPS_S3_ROOT_URL', ''),
    s3_profile=os.getenv('NUPLAN_S3_PROFILE', ''))


class BlobStoreCreator:

    @classmethod
    def create_nuplandb(cls, data_root: str, conf: RemoteConfig = nuplandb_conf, verbose: bool = False) -> BlobStore:
        """
        Create nuPlan DB blob storage.

        :param data_root: nuPlan database root.
        :param conf: Configuration to use, defaults to nuplandb_conf.
        :param verbose: Verbose setting, defaults to False.
        :return: Blob storage created.
        """
        return cls.create(data_root, conf, verbose)

    @classmethod
    def create_mapsdb(cls, map_root: str, conf: RemoteConfig = mapsdb_conf, verbose: bool = False) -> BlobStore:
        """
        Create Maps DB blob storage.

        :param map_root: Maps database root.
        :param conf: Configuration to use, defaults to mapsdb_conf.
        :param verbose: Verbose setting, defaults to False.
        :return: Blob storage created.
        """
        return cls.create(map_root, conf, verbose)

    @classmethod
    def create(cls, data_root: str, conf: RemoteConfig, verbose: bool = False) -> BlobStore:
        """
        Create blob storage.

        :param data_root: Data root.
        :param conf: Configuration to use.
        :param verbose: Verbose setting, defaults to False.
        :return: Blob storage created.
        """
        requested_data_store = os.getenv('NUPLAN_DATA_STORE', 'local')
        cache_on_local_disk = os.getenv('NUPLAN_DO_CACHE', 'true').lower() == "true"

        if requested_data_store == "http":
            # Alerts the user with an exception if we can't access the data server.
            if not conf.http_root_url:
                raise ValueError("Expect http root url to be specified if using http storage.")
            requests.get(conf.http_root_url, timeout=2.0)
            logger.debug(f'Using HTTP blob store {conf.http_root_url} WITH local disk cache at {data_root}')
            # Always use a cache with the HTTPStore to avoid overwhelming the data server.
            return CacheStore(data_root, HttpStore(conf.http_root_url))
        elif requested_data_store == "local":
            logger.debug(f'Using local disk store at {data_root} with no remote store')
            return LocalStore(data_root)
        # Default to S3 if environment variable is empty or not set.
        elif requested_data_store == "s3":
            if not conf.s3_root_url or not conf.s3_profile:
                raise ValueError("Expect the s3 root url and profile to be specified if using s3 storage.")
            store = S3Store(conf.s3_root_url, conf.s3_profile)
            # We don't want to cache on disk for training (there's too much data), but users
            # can set this environment variable if they want to cache data when working locally.
            if cache_on_local_disk:
                logger.debug(f'Using s3 blob store for {conf.s3_root_url} WITH local disk cache at {data_root}')
                return CacheStore(data_root, store)
            else:
                logger.debug(f'Using s3 blob store for {conf.s3_root_url} WITHOUT local disk cache')
                return store
        else:
            raise ValueError(f"Environment variable NUPLAN_DATA_STORE was set to '{requested_data_store}'. "
                             f"Valid values are 'http', 'local', 's3'.")
