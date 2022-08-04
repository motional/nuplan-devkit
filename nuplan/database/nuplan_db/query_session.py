import sqlite3
from typing import Any, Generator, Optional


def execute_many(query_text: str, query_parameters: Any, db_file: str) -> Generator[sqlite3.Row, None, None]:
    """
    Runs a query with the provided arguments on a specified Sqlite DB file.
    This query can return any number of rows.
    :param query_text: The query to run.
    :param query_parameters: The parameters to provide to the query.
    :param db_file: The DB file on which to run the query.
    :return: A generator of rows emitted from the query.
    """
    # Caching a connection saves around 600 uS for local databases.
    # By making it stateless, we get isolation, which is a huge plus.
    connection = sqlite3.connect(db_file)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    try:
        cursor.execute(query_text, query_parameters)

        for row in cursor:
            yield row
    finally:
        cursor.close()
        connection.close()


def execute_one(query_text: str, query_parameters: Any, db_file: str) -> Optional[sqlite3.Row]:
    """
    Runs a query with the provided arguments on a specified Sqlite DB file.
    Validates that the query returns at most one row.
    :param query_text: The query to run.
    :param query_parameters: The parameters to provide to the query.
    :param db_file: The DB file on which to run the query.
    :return: The returned row, if it exists. None otherwise.
    """
    # Caching a connection saves around 600 uS for local databases.
    # By making it stateless, we get isolation, which is a huge plus.
    connection = sqlite3.connect(db_file)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    try:
        cursor.execute(query_text, query_parameters)

        result: Optional[sqlite3.Row] = cursor.fetchone()

        # Check for more rows. If more exist, throw an error.
        if result is not None and cursor.fetchone() is not None:
            raise RuntimeError("execute_one query returned multiple rows.")

        return result
    finally:
        cursor.close()
        connection.close()
