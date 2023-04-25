import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, cast

import pandas as pd

from nuplan.common.utils.io_utils import safe_path_to_string
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


@dataclass(frozen=True)
class ReadMetadataFromS3Input:
    """
    An internal class used to schematize the information needed to parallelize the reading of S3 metadata.
    """

    metadata_filename: str
    cache_path: Path


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
    using_s3_cache_path = str(cache_path).startswith('s3:/')
    # Convert s3 path into proper string format
    sanitized_cache_path = safe_path_to_string(cache_path)

    cache_metadata_storage_path = os.path.join(
        sanitized_cache_path, 'metadata', f'{cache_name}_metadata_node_{node_id}.csv'
    )
    if not using_s3_cache_path:
        Path(cache_metadata_storage_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Using cache_metadata_storage_path: {cache_metadata_storage_path}')
    # The following to_csv function handles both local and s3 paths (via s3fs).
    pd.DataFrame(cache_metadata_entries_dicts).to_csv(cache_metadata_storage_path, index=False)


def _read_metadata_from_s3(inputs: List[ReadMetadataFromS3Input]) -> List[CacheMetadataEntry]:
    """
    Reads metadata csv from s3.
    :param inputs: The inputs to use for the function.
    :returns: The read metadata.
    """
    outputs: List[CacheMetadataEntry] = []
    if len(inputs) == 0:
        return outputs

    sanitized_cache_path = safe_path_to_string(inputs[0].cache_path)
    s3_store = S3Store(sanitized_cache_path)

    for input_value in inputs:
        df = pd.read_csv(s3_store.get(input_value.metadata_filename))
        metadata_dict_list = df.to_dict("records")
        for metadata_dict in metadata_dict_list:
            outputs.append(CacheMetadataEntry(**metadata_dict))

    return outputs


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
    parallel_inputs = [
        ReadMetadataFromS3Input(cache_path=cache_path, metadata_filename=mf) for mf in metadata_filenames
    ]

    result = worker_map(worker, _read_metadata_from_s3, parallel_inputs)
    return cast(List[CacheMetadataEntry], result)


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
