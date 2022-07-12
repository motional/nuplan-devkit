from __future__ import annotations

import abc
import logging
import os
import sqlite3
import threading
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar, Union, overload

import sqlalchemy
from sqlalchemy import event
from sqlalchemy.orm import Session

from nuplan.database.common.blob_store.cache_store import CacheStore
from nuplan.database.common.blob_store.creator import BlobStoreCreator

logger = logging.getLogger(__name__)


T = TypeVar('T')


class DBPathError(Exception):
    """DB Path Error."""

    pass


class DBSplitterInterface(abc.ABC):
    """
    Interface for DB splitters. A DB splitter is responsible for splitting a DB into machine learning
    splits. Splits names are not fixed by this interface and can vary between implementations, but the splits
    themselves are assumed to be defined as a list of DB tokens.
    """

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
        self._engine_pool = defaultdict(dict)  # type: Dict[int, Dict[threading.Thread, sqlalchemy.engine.Engine]]
        # Sessions for each thread, because session can not be shared among multiple threads.
        self._session_pool = defaultdict(dict)  # type: Dict[int, Dict[threading.Thread, Session]]

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
            # Turn on autocommit and autoflush so that the DB transaction history gets flushed to disk.
            # Otherwise, the transaction history will be saved only in memory, leading to unbounded memory growth.
            self._session_pool[pid][t] = Session(bind=self.engine, autocommit=True, autoflush=True)

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
            log = nuplandb.log.select_one(logfile='2021.07.16.20.45.29_veh-35_01095_01486')
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

    def detach(self) -> None:
        """
        Performs any necessary cleanup of the table for destruction.
        This function must be called once the table is ready to be destroyed to properly free resources.
        Once this function is called, the table should no longer be queried.
        """
        # Any event listener registerd with event.load() must be removed manually.
        # Otherwise, SQLAlchemy will keep a reference to the object alive.
        #
        # Unfortunately, this cannot go into __del__, because it will never be called
        #  even if all references to it are removed, because SQLAlchemy will
        #  still hold the reference.
        event.remove(self._table, 'load', self._decorate_record)

    def __len__(self) -> int:
        """
        Return length of the records for the given queries. For example:
            nuplandb.log.__len()
        :return: Number of records.
        """
        return self._session.query(self._table.token).count()  # type: ignore

    @overload
    def __getitem__(self, index: int) -> T:
        """Inherited, see superclass."""
        ...

    @overload  # noqa: F811
    def __getitem__(self, token: str) -> T:
        """Inherited, see superclass."""
        ...

    @overload  # noqa: F811
    def __getitem__(self, indexes: slice) -> List[T]:
        """Inherited, see superclass."""
        ...

    def __getitem__(self, key: Union[int, str, slice]) -> T:  # type: ignore  # noqa: F811
        """Inherited, see superclass."""
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
    Base class for DB loaders. Inherited classes should implement property method for each table with type
    annotation, for example:
        class NuPlanDB(DB):
            @property
            def category(self) -> Table[nuplandb_model.Category]:
                return self.tables['category']

    It is not recommended to use db.get('category', some_token), use db.category.get(some_token) or
    db.category[some_token] instead, because we can't get any type hint from the former one.
    """

    def __init__(
        self,
        table_names: List[str],
        models: Any,
        data_root: str,
        db_path: str,
        verbose: bool,
        model_source_dict: Dict[str, str] = {},
    ):
        """
        Initialize database by loading from filesystem or downloading from S3, load json table and build token index.
        :param table_names: List of table names.
        :param models: Auto-generated model template.
        :param data_root: Path to load the database from; if the database is downloaded from S3
                          this is the path to store the downloaded database.
        :param db_path: Local or S3 path to the database file.
        :param verbose: Whether to print status messages when loading the database.
        """
        self._table_names = list(table_names)
        self._data_root = data_root
        self._blob_store = BlobStoreCreator.create_nuplandb(data_root)
        self._tables = {}
        self._tables_detached = False

        # We have circular references between the DB and the Tables.
        # Because tables use the SqlAlchemy event hooks, they will never be destroyed until the hooks are destroyed.
        #
        # To detect when it is safe to remove the hooks and allow the GC to collect the tables, we must reference count manually.
        # This variable is used by add_ref and remove_ref
        self._refcount = 1
        self._refcount_lock = threading.Lock()

        # Append the correct extension to the filename and prepend the data root if the file does not exist.
        db_path = db_path if db_path.endswith('.db') else f'{db_path}.db'
        self._db_path = Path(db_path)
        self._filename = self._db_path if self._db_path.exists() else Path(self._data_root) / self._db_path.name

        if not self._filename.exists():
            logger.debug(f"DB path not found, downloading db file to {self._filename}...")
            start_time = time.time()
            cache_store = CacheStore(self._data_root, self._blob_store)
            cache_store.save_to_disk(self._db_path.name)
            logger.debug("Downloading db file took {:.1f} seconds".format(time.time() - start_time))

        if verbose:
            logger.debug("\nLoading tables for database {}...".format(self.name))
            start_time = time.time()

        self._session_manager = SessionManager(self._create_db_instance)

        for table_name in self._table_names:
            model_name = ''.join([s.capitalize() for s in table_name.split('_')])
            if len(model_source_dict) != 0:
                if model_name in model_source_dict:
                    model_pcls = getattr(models, model_source_dict[model_name])
                else:
                    model_pcls = getattr(models, model_source_dict['default'])
                model_cls = getattr(model_pcls, model_name)
            else:
                model_cls = getattr(models, model_name)
            self._tables[table_name] = Table[model_cls](model_cls, self)  # type: ignore

        if verbose:
            for table_name in self._table_names:
                logger.debug("{} {},".format(len(self._tables[table_name]), table_name))
            logger.debug("Done loading in {:.1f} seconds.\n".format(time.time() - start_time))

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        return "{}('{}', data_root='{}')".format(self.__class__.__name__, self.name, self.data_root)

    def __str__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        _str = '{} {} with tables:\n{}'.format(self.__class__.__name__, self.name, '=' * 30)
        for table_name in self.table_names:
            if 'log' == table_name:
                continue
            _str += '\n{:20}: {}'.format(table_name, getattr(self, table_name).count())
        return _str

    @property
    def session(self) -> Session:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return self._session_manager.session

    @property
    def name(self) -> str:
        """
        Get the db name.
        :return: The db name.
        """
        return self._db_path.stem

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
        return str(self._filename)

    @property
    def table_names(self) -> List[str]:
        """
        Get the list of table names.
        :return: The list of table names.
        """
        self._assert_tables_attached()
        return self._table_names

    @property
    def tables(self) -> Dict[str, Table[Any]]:
        """
        Get the list of tables.
        :return: The list of tables.
        """
        self._assert_tables_attached()
        return self._tables

    def load_blob(self, path: str) -> BinaryIO:
        """
        Loads a blob.
        :param path: Path to the blob.
        :return: A binary stream to read the blob.
        """
        return self._blob_store.get(path)  # type: ignore

    def get(self, table: str, token: str) -> Any:
        """
        Returns a record from table.
        :param table: Table name.
        :param token: Token of the record.
        :return: The record. See "templates.py" for details.
        """
        warnings.warn("deprecated", DeprecationWarning)
        self._assert_tables_attached()
        return getattr(self, table).get(token)

    def field2token(self, table: str, field: str, query: str) -> List[str]:
        """
        Function returns a list of tokens given a table and field of that table.
        :param table: Table name.
        :param field: Field name, see "template.py" for details.
        :param query: The same type as the field.
        :return: Return a list of record tokens.
        """
        warnings.warn("deprecated", DeprecationWarning)
        self._assert_tables_attached()
        return [rec.token for rec in getattr(self, table).search(**{field: query})]

    def are_tables_detached(self) -> bool:
        """
        Returns true if the tables have been detached, false otherwise.
        :returns: True if the tables have been detached, false otherwise.
        """
        return self._tables_detached

    def detach_tables(self) -> None:
        """
        Prepares all tables for destruction.
        This must be called when DB is ready to be released to reclaim used memory.
        After calling this method, no further queries should be run from the db.

        Placing this in __del__ is not sufficient, because without detaching tables,
          SQLAlchemy will keep references to the tables alive.
          Which contain references to the DB.
          Which means that __del__ will never be called.
        """
        if not self._tables_detached:
            for table_name in self.table_names:
                self.tables[table_name].detach()
            self._tables_detached = True

    def _assert_tables_attached(self) -> None:
        """
        Checks to ensure that the tables are attached. If not, raises an error.
        """
        if self.are_tables_detached():
            raise RuntimeError("Attempting to query from detached tables.")

    def add_ref(self) -> None:
        """
        Add an external reference to this class to prevent it from being reclaimed by the GC.
        This method should be called when any non-SqlAlchemy class takes a reference to the class.

        See the comments in __init__ for explanation
        """
        with self._refcount_lock:
            # We don't have a great way of reattaching tables.
            # If someone really needs this, they should create a new db object entirely.
            if self._refcount == 0:
                raise ValueError(
                    "Attempting to revive a database that has had its tables detached. This is likely due to a reference counting error."
                )
            self._refcount += 1

    def remove_ref(self) -> None:
        """
        Removes an external reference to this class.
        This should be called when any non-SqlAlchemy class is finished using the database (e.g. in their __del__ method).
        If the reference count gets to zero, it will be prepared for collection by the GC.
        """
        with self._refcount_lock:
            self._refcount -= 1
            if self._refcount == 0:
                self.detach_tables()

    def _create_db_instance(self) -> sqlite3.Connection:
        """
        Internal method, return sqlite3 connection for sqlalchemy.
        :return: Sqlite3 connection.
        """
        assert Path(self.table_root).exists(), 'DB file not found: {}'.format(self.table_root)
        db = sqlite3.connect('file:{}?mode=ro'.format(self.table_root), uri=True, check_same_thread=False)
        db.execute('PRAGMA main.journal_mode = OFF;')
        db.execute('PRAGMA main.cache_size=10240;')
        db.execute('PRAGMA main.page_size = 4096;')
        db.execute('PRAGMA main.journal_mode = OFF;')
        db.execute('PRAGMA query_only = 1;')
        return db
