import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadataEntry:
    """
    Metadata for cached model input features.
    """

    file_name: Path


@dataclass
class CacheResult:
    """
    Results returned from caching model input features. Includes number of sucessfully cached features,
    number of failures and cache metadata entries.
    """

    successes: int
    failures: int
    cache_metadata: List[Optional[CacheMetadataEntry]]


def save_cache_metadata(cache_metadata_entries: List[CacheMetadataEntry], cache_path: Path, node_id: int) -> None:
    """
    Saves list of CacheMetadataEntry to output csv file path.
    :param cache_metadata_entries: List of metadata objects for cached features.
    :param cache_path: Path to s3 cache.
    :param node_id: Node ID of a node used for differentiating between nodes in multi-node caching.
    """
    # Convert list of dataclasses into list of dictionaries
    cache_metadata_entries_dicts = [asdict(entry) for entry in cache_metadata_entries]
    cache_name = cache_path.name

    # Convert s3 path into proper string format
    sanitised_cache_path = sanitise_s3_path(cache_path)
    cache_metadata_storage_path = f'{sanitised_cache_path}/metadata/{cache_name}_metadata_node_{node_id}.csv'
    logger.info(f'Using cache_metadata_storage_path: {cache_metadata_storage_path}')

    pd.DataFrame(cache_metadata_entries_dicts).to_csv(cache_metadata_storage_path, index=False)


def read_cache_metadata(
    cache_path: Path, metadata_filenames: List[str], worker: WorkerPool
) -> List[CacheMetadataEntry]:
    """
    Reads csv file path into list of CacheMetadataEntry.
    :param cache_path: Path to s3 cache.
    :param metadata_filenames: Filenames of the metadata csv files.
    :return: List of CacheMetadataEntry.
    """
    # Convert s3 path into proper string format
    sanitised_cache_path = sanitise_s3_path(cache_path)
    s3_store = S3Store(sanitised_cache_path)
    metadata_dataframes = [pd.read_csv(s3_store.get(filename)) for filename in metadata_filenames]
    metadata_dicts = [metadata_dict for df in metadata_dataframes for metadata_dict in df.to_dict('records')]
    cache_metadata_entries = worker_map(worker, _construct_cache_metadata_entry_from_dict, metadata_dicts)
    return cast(List[CacheMetadataEntry], cache_metadata_entries)


def _construct_cache_metadata_entry_from_dict(metadata_dicts: List[Dict[str, Any]]) -> List[CacheMetadataEntry]:
    """
    Constructs CacheMetadataEntry from list of metadata_dicts
    :param metadata_dicts: List of metadata dictionaries.
    :return: List of CacheMetadataEntry
    """
    cache_metadata_entries = [CacheMetadataEntry(**metadata_dict) for metadata_dict in metadata_dicts]
    return cache_metadata_entries


def sanitise_s3_path(s3_path: Path) -> str:
    """
    Sanitises s3 paths from Path objects to string.
    :param s3_path: Path object of s3 path
    :return: s3 path with the correct format as a string.
    """
    return f's3://{str(s3_path).lstrip("s3:/")}'


def extract_field_from_cache_metadata_entries(
    cache_metadata_entries: List[CacheMetadataEntry], desired_attribute: str
) -> List[Any]:
    """
    Extracts specified field from cache metadata entries.
    :param cache_metadata_entries: List of CacheMetadataEntry
    :return: List of desired attributes in each CacheMetadataEntry
    """
    metadata = [getattr(entry, desired_attribute) for entry in cache_metadata_entries]
    return metadata
