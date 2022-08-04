from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from tqdm import tqdm

from nuplan.common.utils.s3_utils import check_s3_path_exists, expand_s3_dir
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.scene import Scene

logger = logging.getLogger(__name__)


MAX_DB_LOADING_THREADS = 64  # Maximum number of threads to be used when loading the databases


def discover_log_dbs(load_path: Union[List[str], str]) -> List[str]:
    """
    Discover all log dbs from the input load path.
    If the path is a filename, expand the path and return the list of filenames in that path.
    Else, if the path is already a list, expand each path in the list and return the flattened list.
    :param load_path: Load path, it can be a filename or list of filenames of a database and/or dirs of databases.
    :return: A list with all discovered log database filenames.
    """
    if isinstance(load_path, list):  # List of database paths
        nested_db_filenames = [get_db_filenames_from_load_path(path) for path in sorted(load_path)]
        db_filenames = [filename for filenames in nested_db_filenames for filename in filenames]
    else:
        db_filenames = get_db_filenames_from_load_path(load_path)

    return db_filenames


def get_db_filenames_from_load_path(load_path: str) -> List[str]:
    """
    Retrieve all log database filenames from a load path.
    The path can be either local or remote (S3).
    The path can represent either a single database filename (.db file) or a directory containing files.
    :param load_path: Load path, it can be a filename or list of filenames.
    :return: A list of all discovered log database filenames.
    """
    if load_path.endswith('.db'):  # Single database path
        if load_path.startswith('s3://'):  # File is remote (S3)
            assert check_s3_path_exists(load_path), f'S3 db path does not exist: {load_path}'
            os.environ['NUPLAN_DATA_ROOT_S3_URL'] = load_path.rstrip(Path(load_path).name)
        else:  # File is local
            assert Path(load_path).is_file(), f'Local db path does not exist: {load_path}'
        db_filenames = [load_path]
    else:  # Path to directory containing databases
        if load_path.startswith('s3://'):  # Directory is remote (S3)
            db_filenames = expand_s3_dir(load_path, filter_suffix='.db')
            assert len(db_filenames) > 0, f'S3 dir does not contain any dbs: {load_path}'
            os.environ['NUPLAN_DATA_ROOT_S3_URL'] = load_path  # TODO: Deprecate S3 data root env variable
        elif Path(load_path).expanduser().is_dir():  # Directory is local
            db_filenames = [
                str(path) for path in sorted(Path(load_path).expanduser().iterdir()) if path.suffix == '.db'
            ]
        else:
            raise ValueError(f'Expected db load path to be file, dir or list of files/dirs, but got {load_path}')

    return db_filenames


def load_log_db_mapping(
    data_root: str,
    db_files: Optional[Union[List[str], str]],
    maps_db: GPKGMapsDB,
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, NuPlanDB]:
    """
    Load all log database objects and hash them based on their log name.
    Log databases will be discovered based on the input path and downloaded if needed.
    All discovered databases will be loaded/downloaded concurrently in multiple threads.
    :param data_root: Local data root for loading/storing the log databases.
                      If `db_files` is not None, all downloaded databases will be stored to this data root.
    :param db_files: Local/remote filename or list of filenames to be loaded.
                     If None, all database filenames found under `data_root` will be used.
    :param maps_db: Instantiated map database object to be passed to each log database.
    :param max_workers: Maximum number of workers to use when loading the databases concurrently.
                        Only used when the number of databases to load is larger than this parameter.
    :param verbose: Whether to print progress and details during the database loading process.
    :return: Mapping from log name to loaded log database object.
    """
    # Discover all database filenames to be loaded.
    load_path = data_root if db_files is None else db_files
    db_filenames = discover_log_dbs(load_path)

    # Decide number of workers to use for creating the databases.
    num_workers = min(len(db_filenames), MAX_DB_LOADING_THREADS)
    num_workers = num_workers if max_workers is None else min(num_workers, max_workers)

    # Use multi-threading for creating all databases as loading/downloading are primarily IO-bound tasks.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(NuPlanDB, data_root, path, maps_db, verbose) for path in db_filenames]
        futures_iterable = as_completed(futures)
        wrapped_iterable = tqdm(futures_iterable, total=len(futures), leave=False) if verbose else futures_iterable
        log_dbs = [future.result() for future in wrapped_iterable]

    # Sort databases based on their log name and hash them in a table.
    log_dbs = sorted(log_dbs, key=lambda log_db: str(log_db.log_name))
    log_db_mapping = {log_db.log_name: log_db for log_db in log_dbs}

    return log_db_mapping


class NuPlanDBWrapper:
    """Wrapper for NuPlanDB that allows loading and accessing mutliple log database."""

    def __init__(
        self,
        data_root: str,
        map_root: str,
        db_files: Optional[Union[List[str], str]],
        map_version: str,
        max_workers: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize the database wrapper.
        :param data_root: Local data root for loading (or storing downloaded) the log databases.
                        If `db_files` is not None, all downloaded databases will be stored to this data root.
        :param map_root: Local map root for loading (or storing downloaded) the map database.
        :param db_files: Path to load the log databases from, which can be:
                         * filename path to single database:
                            - locally - e.g. /data/sets/nuplan/v1.0/log_1.db
                            - remotely (S3) - e.g. s3://bucket/nuplan/v1.0/log_1.db
                         * directory path of databases to load:
                            - locally - e.g. /data/sets/nuplan/v1.0/
                            - remotely (S3) - e.g. s3://bucket/nuplan/v1.0/
                         * list of database filenames:
                            - locally - e.g. [/data/sets/nuplan/v1.0/log_1.db, /data/sets/nuplan/v1.0/log_2.db]
                            - remotely (S3) - e.g. [s3://bucket/nuplan/v1.0/log_1.db, s3://bucket/nuplan/v1.0/log_2.db]
                         * list of database directories:
                            - locally - e.g. [/data/sets/nuplan/v1.0_split_1/, /data/sets/nuplan/v1.0_split_2/]
                            - remotely (S3) - e.g. [s3://bucket/nuplan/v1.0_split_1/, s3://bucket/nuplan/v1.0_split_2/]
                         Note: Regex expansion is not yet supported.
                         Note: If None, all database filenames found under `data_root` will be used.
        :param map_version: Version of map database to load. The map database is passed to each loaded log database.
        :param max_workers: Maximum number of workers to use when loading the databases concurrently.
                            Only used when the number of databases to load is larger than this parameter.
        :param verbose: Whether to print progress and details during the database loading process.
        """
        self._data_root = data_root
        self._map_root = map_root
        self._db_files = db_files
        self._map_version = map_version
        self._max_workers = max_workers
        self._verbose = verbose

        # Data and map root paths must be in the local filesystem
        assert not self._data_root.startswith('s3://'), f'Data root cannot be an S3 path, got {self._data_root}'
        assert not self._map_root.startswith('s3://'), f'Map root cannot be an S3 path, got {self._map_root}'

        # Fix path strings loaded from hydra
        self._data_root = self._data_root.replace('//', '/')
        self._map_root = self._map_root.replace('//', '/')

        # Load maps DB
        self._maps_db = GPKGMapsDB(map_root=self._map_root, map_version=self._map_version)
        logger.info('Loaded maps DB')

        # Load nuPlan log DBs
        self._log_db_mapping = load_log_db_mapping(
            data_root=self._data_root,
            db_files=self._db_files,
            maps_db=self._maps_db,
            max_workers=self._max_workers,
            verbose=self._verbose,
        )
        logger.info(f'Loaded {len(self.log_dbs)} log DBs')

    def __reduce__(self) -> Tuple[Type[NuPlanDBWrapper], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (
            self._data_root,
            self._map_root,
            self._db_files,
            self._map_version,
            self._max_workers,
            self._verbose,
        )

    def __del__(self) -> None:
        """
        Called when the object is being garbage collected.
        """
        # Remove this object's reference to the included tables.
        for log_name in self.log_names:
            self._log_db_mapping[log_name].remove_ref()

    @property
    def data_root(self) -> str:
        """Get the data root."""
        return self._data_root

    @property
    def map_root(self) -> str:
        """Get the map root."""
        return self._map_root

    @property
    def map_version(self) -> str:
        """Get the map version."""
        return self._map_version

    @property
    def maps_db(self) -> GPKGMapsDB:
        """Get the map database object."""
        return self._maps_db

    @property
    def log_db_mapping(self) -> Dict[str, NuPlanDB]:
        """Get the dictionary that maps log names to log database objects."""
        return self._log_db_mapping

    @property
    def log_names(self) -> List[str]:
        """Get the list of log names of all loaded log databases."""
        return list(self._log_db_mapping.keys())

    @property
    def log_dbs(self) -> List[NuPlanDB]:
        """Get the list of all loaded log databases."""
        return list(self._log_db_mapping.values())

    def get_log_db(self, log_name: str) -> NuPlanDB:
        """
        Retrieve a log database by log name.
        :param log_name: Log name to access the database hash table.
        :return: Retrieve database object.
        """
        return self._log_db_mapping[log_name]

    def get_all_scenes(self) -> Iterable[Scene]:
        """
        Retrieve and yield all scenes across all loaded log databases.
        :yield: Next scene from all scenes in the loaded databases.
        """
        for db in self._log_db_mapping.values():
            for scene in db.scene:
                yield scene

    def get_all_scenario_types(self) -> List[str]:
        """Retrieve all unique scenario tags in the collection of databases."""
        return sorted({tag for log_db in self.log_dbs for tag in log_db.get_unique_scenario_tags()})
