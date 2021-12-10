from __future__ import annotations

import abc
import logging
import os
import os.path as osp
import sqlite3
import threading
import time
import warnings
from collections import defaultdict
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar, Union, overload

import sqlalchemy
from nuplan.database.common.blob_store.blob_store import BlobStore
from nuplan.database.common.blob_store.cache_store import CacheStore
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.maps_db.imapsdb import IMapsDB
from sqlalchemy import event
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DBPathError(Exception):
    """ DB Path Error. """
    pass


class DBSplitterInterface(abc.ABC):
    """ Interface for DB splitters. A DB splitter is responsible for splitting a DB into machine learning
     splits. Splits names are not fixed by this interface and can vary between implementations, but the splits
     themselves are assumed to be defined as a list of DB tokens. """

    def __str__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        out = 'Splits:\n{}\n'.format('-' * 28)
        for split_name in self.list():
            out += '{:20s}: {:>6d}\n'.format(split_name, len(self.split(split_name)))
        return out

    @abc.abstractmethod
    def list(self) -> List[str]:
        """
        Return the list of all split names.
        :return: The list of all split names.
        """
        pass

    @abc.abstractmethod
    def split(self, split_name: str) -> List[str]:
        """
        Return record tokens in the split given by split_name.
        :return: The list of record tokens in the given split.
        """
        pass

    @abc.abstractmethod
    def logs(self, split_name: str) -> List[str]:
        """
        Return the list of log names present in split split_name.
        :return: The list of log names in the given split.
        """
        pass


class SessionManager:
    """
    We use this to support multi-processes/threads. The idea is to have one
    db connection for each process, and have one session for each thread.
    """

    def __init__(self, engine_creator: Callable[[], Any]) -> None:
        """
        :param engine_creator: A callable which returns a DBAPI connection.
        """
        self._creator = engine_creator

        # Engines for each thread, because engine can not be shared among multiple threads.
        self._engine_pool = defaultdict(dict)       # type: Dict[int, Dict[threading.Thread, sqlalchemy.engine.Engine]]
        # Sessions for each thread, because session can not be shared among multiple threads.
        self._session_pool = defaultdict(dict)      # type: Dict[int, Dict[threading.Thread, Session]]

    @property
    def engine(self) -> sqlalchemy.engine.Engine:
        """
        Get the engine for the current thread. A new one will be created if not already exist.
        :return: The underlying engine.
        """
        pid = os.getpid()
        t = threading.current_thread()

        if t not in self._engine_pool[pid]:
            self._engine_pool[pid][t] = sqlalchemy.create_engine('sqlite:///', creator=self._creator)

        return self._engine_pool[pid][t]

    @property
    def session(self) -> Session:
        """
        Get the session for the current thread. A new one will be created if not already exist.
        :return: The underlying session.
        """
        pid = os.getpid()
        t = threading.current_thread()

        if t not in self._session_pool[pid]:
            self._session_pool[pid][t] = Session(bind=self.engine, autocommit=False, autoflush=False)

        return self._session_pool[pid][t]


class Table(Sequence[T]):
    """
    Table wrapper. Provide some convenient APIs, for example:
        table = Table(Sample, session)

        first_row = table[0]
        last_row = table[-1]
        some_random_rows = table[50:100]
        my_row = table['row_token_here']

        total_num = len(table)
    """

    def __init__(self, table: Any, db: DB) -> None:
        """
        Init table.
        :param table: Class type in models.py.
        :param db: DB instance.
        """
        self._table = table
        # This used to be a weakref, but we once saw a "ReferenceError: weakly-referenced
        # object no longer exists" error on this line in the Image.load_bytes_jpg method:
        #     return self.table.db.load_blob(self.filename_jpg)
        # Making this a 'strong' reference means that Table and DB instance can't be
        # garbage collected until all the records have been garbage collected, but the
        # Table and DB instances aren't huge, and modifying the code to handle a weakref
        # becoming invalid might require some larger refactoring.
        self._db = db
        event.listen(table, 'load', self._decorate_record)

    def _decorate_record(self, record: T, context: Any) -> None:
        """
        Sqlalchemy hook function. This will be called each time sqlalchemy loads a object from db.
        We save table reference as "_table" in the record here.
        :param record: The record loaded from database.
        :param context: Some context we don't use.
        """
        # This used to be a weakref, see comment in __init__.
        record._table = self  # type: ignore

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        count = self.count()
        _repr = str(self._table.__name__) + '({} entries):\n{}\n'.format(count, '-' * 50)

        for ind in range(count)[:3]:
            _repr += repr(self._session.query(self._table)[ind])

        if count > 3:
            _repr += '(...) \n'
            _repr += repr(self._session.query(self._table)[count - 1])

        return _repr

    @property
    def _session(self) -> Session:
        """
        Get the underlying db session.
        :return: The underlying db session.
        """
        return self.db.session

    @property
    def db(self) -> DB:
        """
        Get the underlying db.
        :return: The underlying db.
        """
        return self._db

    def get(self, token: str) -> T:
        """
        Returns a record from table.
        :param token: Token of the record.
        :return: Record object.
        """
        return self._session.query(self._table).get(token)  # type: ignore

    def select_one(self, **filters: Any) -> Optional[T]:
        """
        Query table using filters. There should be at most one record matching
        given filters, use select_many for searching multiple records.
            cat = nuplandb.category.select_one(name='vehicle')
        :param filters: Query using keyword expression. For example, query log by log file name:
            log = nuplandb.log.select_one(logfile='2021.05.26.20.05.14_38_1622073985538950.8_1622074969538793.5')
        :return: Record object matching the given filters.
        """
        record: Optional[T] = self._session.query(self._table).filter_by(**filters).one_or_none()
        return record

    def select_many(self, **filters: Any) -> List[T]:
        """
        Query table using filters.
            boston_logs = nuplandb.log.select_many(location='boston').
        :param filters: Query using keyword expression. For example, query log by vehicle:
            logs = nuplandb.log.select_many(vehicle_name='35')
        :return: A list of records mathing the given filters.
        """
        return self._session.query(self._table).filter_by(**filters).all()  # type: ignore

    def count(self, **kwargs: Any) -> int:  # type: ignore
        """
        Count records for the given queries. For example:
            nuplandb.log.count(location='las_vegas').
        :param kwargs: Filters to count records.
        :return: The number of counted records.
        """
        return self._session.query(self._table).filter_by(**kwargs).count()  # type: ignore

    def all(self) -> List[T]:
        """
        Return all the items for the given queries. For example:
            nuplandb.log.all().
        :return: List of records.
        """
        return self._session.query(self._table).all()  # type: ignore

    def __len__(self) -> int:
        """
        Return length of the records for the given queries. For example:
            nuplandb.log.__len()
        :return: Number of records.
        """
        return self._session.query(self._table.token).count()  # type: ignore

    @overload
    def __getitem__(self, index: int) -> T:
        """ Inherited, see superclass. """
        ...

    @overload  # noqa: F811
    def __getitem__(self, token: str) -> T:
        """ Inherited, see superclass. """
        ...

    @overload  # noqa: F811
    def __getitem__(self, indexes: slice) -> List[T]:
        """ Inherited, see superclass. """
        ...

    def __getitem__(self, key: Union[int, str, slice]) -> T:  # type: ignore # noqa: F811
        """ Inherited, see superclass. """
        if isinstance(key, str):
            return self._session.query(self._table).get(key)  # type: ignore
        else:
            return self._session.query(self._table)[key]  # type: ignore

    def __iter__(self) -> Iterator[T]:
        """
        Inherited, see superclass.
        The implementation in Sequence is not good, it uses self[i] to loop over all elements
        which makes it much slower.
        Detail of yield_per can be found:
            https://docs.sqlalchemy.org/en/latest/orm/query.html#sqlalchemy.orm.query.Query.yield_per
        """
        query = self._session.query(self._table).yield_per(2000).enable_eagerloads(False)
        for row in query:
            yield row


class DB:
    """
    Base class for DB loaders. Inherited class should implement property method for each table with type
    annotation, for example:

        class NuPlanDB(DB):
            @property
            def category(self) -> Table[nuplandb_model.Category]:
                return self.tables['category']

    It is not recommended to use db.get('category', some_token), use db.category.get(some_token) or
    db.category[some_token] instead, because we can't get any type hint from the former one.
    """

    def __init__(self, table_names: List[str], models: Any, data_root: str, version: str,
                 verbose: bool, blob_store: Optional[BlobStore] = None, maps_db: Optional[IMapsDB] = None):
        """
        Initialize database, load json table and build token index.
        :param table_names: Table names.
        :param models: models.py generated from templates.py.
        :param data_root: Path to the root directory for all versions of the database tables and blobs.
        :param version: Database version.
        :param verbose: Whether to print status messages during load.
        :param blob_store: The blob store db will use to load file blobs.
        :param maps_db: maps database.
        """

        self._table_names = table_names
        self._version = version
        self._data_root = data_root
        self._table_root = osp.join(data_root, version)
        self._tables = {}
        self._blob_store = blob_store or LocalStore(data_root)
        self._maps_db = maps_db

        db_filepath = self._table_root + '.db'
        if not osp.exists(db_filepath):
            logger.debug(f"Downloading db file to {db_filepath}...")
            start_time = time.time()
            cache_store = CacheStore(data_root, blob_store)
            cache_store.save_to_disk(version + '.db')
            logger.debug("Downloading db file took {:.1f} seconds".format(time.time() - start_time))

        start_time = time.time()
        if verbose:
            logger.debug("======\nLoading tables for version {}...".format(self.version))

        self._session_manager = SessionManager(self._create_db_instance)

        for table_name in table_names:
            model_name = ''.join([s.capitalize() for s in table_name.split('_')])
            model_cls = getattr(models, model_name)

            self._tables[table_name] = Table[model_cls](model_cls, self)  # type: ignore

        if verbose:
            for table_name in self._table_names:
                logger.debug("{} {},".format(len(self._tables[table_name]), table_name))
            logger.debug("Done loading in {:.1f} seconds.\n======".format(time.time() - start_time))

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        return "{}('{}', data_root='{}')".format(self.__class__.__name__, self.version, self.data_root)

    def __str__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        _str = '{} {} with tables:\n{}'.format(self.__class__.__name__, self.version, '=' * 30)
        for table_name in self.table_names:
            _str += '\n{:20}: {}'.format(table_name, getattr(self, table_name).count())
        return _str

    @property
    def maps_db(self) -> Optional[IMapsDB]:
        """
        Get the underlying db.
        :return: The underlying db.
        """
        return self._maps_db

    @property
    def session(self) -> Session:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return self._session_manager.session

    @property
    def version(self) -> str:
        """
        Get the db version.
        :return: The db version.
        """
        return self._version

    @property
    def data_root(self) -> str:
        """
        Get the data root.
        :return: The data root.
        """
        return self._data_root

    @property
    def table_root(self) -> str:
        """
        Get the table root.
        :return: The table root.
        """
        return self._table_root

    @property
    def table_names(self) -> List[str]:
        """
        Get the list of table names.
        :return: The list of table names.
        """
        return self._table_names

    @property
    def tables(self) -> Dict[str, Table[Any]]:
        """
        Get the list of tables.
        :return: The list of tables.
        """
        return self._tables

    def load_blob(self, path: str) -> BinaryIO:
        """
        Loads a blob.
        :param path: Path to the blob.
        :return: A binary stream to read the blob.
        """
        return self._blob_store.get(path)   # type: ignore

    def get(self, table: str, token: str) -> Any:
        """
        Returns a record from table.
        :param table: Table name.
        :param token: Token of the record.
        :return: The record. See "templates.py" for details.
        """
        warnings.warn("deprecated", DeprecationWarning)
        return getattr(self, table).get(token)

    def field2token(self, table: str, field: str, query: str) -> List[str]:
        """
        This function returns a list of tokens given a table and field of that table.
        :param table: Table name.
        :param field: Field name, see "template.py" for details.
        :param query: The same type as the field.
        :return: Return a list of record tokens.
        """
        warnings.warn("deprecated", DeprecationWarning)
        return [rec.token for rec in getattr(self, table).search(**{field: query})]

    def _create_db_instance(self) -> sqlite3.Connection:
        """
        Internal method, return sqlite3 connection for sqlalchemy.
        :return: Sqlite3 connection.
        """
        db_file = osp.join(self._data_root, self._version + '.db')
        assert osp.exists(db_file), 'DB file not found: {}'.format(db_file)
        db = sqlite3.connect('file:{}?mode=ro'.format(db_file), uri=True, check_same_thread=False)
        db.execute('PRAGMA main.journal_mode = OFF;')
        db.execute('PRAGMA main.cache_size=10240;')
        db.execute('PRAGMA main.page_size = 4096;')
        db.execute('PRAGMA main.journal_mode = OFF;')
        db.execute('PRAGMA query_only = 1;')
        return db
