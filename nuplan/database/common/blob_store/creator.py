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

RemoteConfig = NamedTuple("RemoteConfig", [("http_root_url", Optional[str]), ("s3_root_url", Optional[str])])

NUPLAN_DATA_STORE = os.getenv("NUPLAN_DATA_STORE", "local")
NUPLAN_CACHE_FROM_S3 = os.getenv("NUPLAN_CACHE_FROM_S3", "true").lower() == "true"


class BlobStoreCreator:
    """BlobStoreCreator Class."""

    @classmethod
    def create_nuplandb(cls, data_root: str, verbose: bool = False) -> BlobStore:
        """
        Create nuPlan DB blob storage.

        :param data_root: nuPlan database root.
        :param verbose: Verbose setting, defaults to False.
        :return: Blob storage created.
        """
        conf = RemoteConfig(
            http_root_url=os.getenv("NUPLAN_DATA_ROOT_HTTP_URL", ""),
            s3_root_url=os.getenv("NUPLAN_DATA_ROOT_S3_URL", ""),
        )

        return cls.create(data_root, conf, verbose)

    @classmethod
    def create_mapsdb(cls, map_root: str, verbose: bool = False) -> BlobStore:
        """
        Create Maps DB blob storage.

        :param map_root: Maps database root.
        :param verbose: Verbose setting, defaults to False.
        :return: Blob storage created.
        """
        conf = RemoteConfig(
            http_root_url=os.getenv("NUPLAN_MAPS_ROOT_HTTP_URL", ""),
            s3_root_url=os.getenv("NUPLAN_MAPS_ROOT_S3_URL", ""),
        )

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
        if NUPLAN_DATA_STORE == "http":
            # Alerts the user with an exception if we can't access the data server.
            if not conf.http_root_url:
                raise ValueError("HTTP root url to be specified if using http storage.")
            requests.get(conf.http_root_url, timeout=2.0)
            logger.debug(f"Using HTTP blob store {conf.http_root_url} WITH local disk cache at {data_root}")
            # Always use a cache with the HTTPStore to avoid overwhelming the data server.
            return CacheStore(data_root, HttpStore(conf.http_root_url))
        elif NUPLAN_DATA_STORE == "local":
            logger.debug(f"Using local disk store at {data_root} with no remote store")
            return LocalStore(data_root)
        # Default to S3 if environment variable is empty or not set.
        elif NUPLAN_DATA_STORE == "s3":
            if not conf.s3_root_url:
                raise ValueError("S3 root url to be specified if using s3 storage. " f"s3_root_url: {conf.s3_root_url}")
            store = S3Store(conf.s3_root_url, show_progress=verbose)
            # We don't want to cache on disk for training (there's too much data), but users
            # can set this environment variable if they want to cache data when working locally.
            if NUPLAN_CACHE_FROM_S3:
                logger.debug(f"Using s3 blob store for {conf.s3_root_url} WITH local disk cache at {data_root}")
                return CacheStore(data_root, store)
            else:
                logger.debug(f"Using s3 blob store for {conf.s3_root_url} WITHOUT local disk cache")
                return store
        else:
            raise ValueError(
                f"Environment variable NUPLAN_DATA_STORE was set to '{NUPLAN_DATA_STORE}'. "
                f"Valid values are 'http', 'local', 's3'."
            )
