import sqlite3
from typing import Any, Generator, Optional
from collections import OrderedDict

memory_dbs = OrderedDict()
MAX_CACHE_SIZE = 5  # 允许的最大缓存连接数

def get_or_copy_db_to_memory(db_file: str) -> sqlite3.Connection:
    """
    Get an existing in-memory database connection or copy the SQLite DB file to an in-memory database if not exists.
    Manages cache size to not exceed MAX_CACHE_SIZE by removing the least recently used (LRU) connection.
    :param db_file: The DB file to check or copy to memory.
    :return: A connection to the in-memory database.
    """
    # 如果已缓存，则将其移动到字典的末尾以标记为最近使用
    print("memory dbs: {}, current,{}".format(memory_dbs,db_file) )
    if db_file in memory_dbs:
        memory_dbs.move_to_end(db_file)
        return memory_dbs[db_file]

    # 如果达到最大缓存大小，则删除最早的项
    if len(memory_dbs) >= MAX_CACHE_SIZE:
        oldest_db_file, oldest_conn = memory_dbs.popitem(last=False)  # 删除第一个添加的项
        oldest_conn.close()
        print(f"Closed and removed the oldest DB from cache: {oldest_db_file}")

    # 创建新的内存数据库连接
    disk_connection = sqlite3.connect(db_file)
    mem_connection = sqlite3.connect(':memory:')
    disk_connection.backup(mem_connection)  # Requires Python 3.7+
    disk_connection.close()

    # 添加到缓存并返回
    memory_dbs[db_file] = mem_connection
    return mem_connection


def execute_many(query_text: str, query_parameters: Any, db_file: str, use_mem = True) -> Generator[sqlite3.Row, None, None]:
    """
    Runs a query on a specified Sqlite DB file, preferably using an in-memory copy for improved speed.
    :param query_text: The query to run.
    :param query_parameters: The parameters to provide to the query.
    :param db_file: The DB file to use, copying to memory if not already done.
    :return: A generator of rows emitted from the query.
    """
    if use_mem:
        connection = get_or_copy_db_to_memory(db_file)
    else:
        connection = sqlite3.connect(db_file)

    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    try:
        cursor.execute(query_text, query_parameters)
        for row in cursor:
            yield row
    finally:
        cursor.close()
        # Do not close the connection here to reuse it

def execute_one(query_text: str, query_parameters: Any, db_file: str) -> Optional[sqlite3.Row]:
    """
    Runs a query on a specified Sqlite DB file, preferably using an in-memory copy for improved speed.
    :param query_text: The query to run.
    :param query_parameters: The parameters to provide to the query.
    :param db_file: The DB file to use, copying to memory if not already done.
    :return: The returned row, if it exists. None otherwise.
    """
    connection = get_or_copy_db_to_memory(db_file)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    try:
        cursor.execute(query_text, query_parameters)
        result: Optional[sqlite3.Row] = cursor.fetchone()
        if result is not None and cursor.fetchone() is not None:
            raise RuntimeError("execute_one query returned multiple rows.")
        return result
    finally:
        cursor.close()
        # Do not close the connection here to reuse it
