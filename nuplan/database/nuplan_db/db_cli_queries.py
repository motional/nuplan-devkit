from typing import Dict, Generator, Tuple

from nuplan.database.nuplan_db.db_description_types import ColumnDescription, DbDescription, TableDescription
from nuplan.database.nuplan_db.query_session import execute_many, execute_one


def _get_table_columns_from_db(log_file: str, table_name: str) -> Generator[ColumnDescription, None, None]:
    """
    Get information about the columns that are present in the table.
    If the table does not exist, returns an empty generator.
    :param log_file: The log file to query.
    :param table_name: The table name to query.
    :return: A generator containing information about the columns in the table, ordered by column_id ascending.
    """
    # Command substitution does not work for PRAGMA statements :(
    query = f"""
    PRAGMA table_info({table_name});
    """

    for row in execute_many(query, (), log_file):
        yield ColumnDescription(
            column_id=row["cid"],
            name=row["name"],
            data_type=row["type"],
            nullable=not row["notnull"],
            is_primary_key=row["pk"],
        )


def _get_table_row_count_from_db(log_file: str, table_name: str) -> int:
    """
    Get the number of rows in a table.
    Raises an error if the table does not exist.
    :param log_file: The log file to query.
    :param table_name: The table name to examine.
    :return: The number of rows in the table.
    """
    # Parameter substitution cannot be used for table names :(
    query = f"""
    SELECT COUNT(*) AS cnt
    FROM {table_name};
    """

    result = execute_one(query, (), log_file)
    if result is None:
        raise ValueError(f"Table {table_name} does not exist.")

    return int(result["cnt"])


def _get_table_description(log_file: str, table_name: str) -> TableDescription:
    """
    Get a description of the table.
    :param log_file: The log file to query.
    :param table_name: The table name to examine.
    :return: A struct filled with information about the table.
    """
    return TableDescription(
        name=table_name,
        columns={tc.name: tc for tc in _get_table_columns_from_db(log_file, table_name)},
        row_count=_get_table_row_count_from_db(log_file, table_name),
    )


def _get_table_names_from_db(log_file: str) -> Generator[str, None, None]:
    """
    Get the names of tables in the DB.
    :param log_file: The log file to examine.
    :return: A generator containing the table names.
    """
    query = """
    SELECT tbl_name
    FROM sqlite_schema
    WHERE type='table'
    ORDER BY tbl_name ASC;
    """

    for row in execute_many(query, (), log_file):
        yield row["tbl_name"]


def get_db_description(log_file: str) -> DbDescription:
    """
    Get information about all tables that are present in the DB.
    :param log_file: The log file to describe.
    :return: A description of the tables present in the DB.
    """
    tables: Dict[str, TableDescription] = {}
    for table_name in _get_table_names_from_db(log_file):
        tables[table_name] = _get_table_description(log_file, table_name)

    return DbDescription(tables=tables)


def get_db_duration_in_us(log_file: str) -> int:
    """
    Get the duration of the database log in us, measured as (last_lidar_pc_timestamp) - (first_lidarpc_timestamp)
    :param log_file: The log file to query.
    :return: The db duration, in microseconds.
    """
    query = """
    SELECT MAX(timestamp) - MIN(timestamp) AS diff_us
    FROM lidar_pc;
    """

    result = execute_one(query, (), log_file)
    return int(result["diff_us"])


def get_db_log_duration(log_file: str) -> Generator[Tuple[str, int], None, None]:
    """
    Get the duration of each log present in the database, measured as (last_lidar_pc_timestamp) - (first_lidarpc_timestamp)
    :param log_file: The log file to query.
    :return: A tuple of (log_name, duration) pair, one for each log file present in the DB, sorted by log name.
    """
    query = """
    SELECT  l.logfile,
            MAX(lp.timestamp) - MIN(lp.timestamp) AS duration_us
    FROM log AS l
    INNER JOIN scene AS s
        ON s.log_token = l.token
    INNER JOIN lidar_pc AS lp
        ON lp.scene_token = s.token
    GROUP BY l.logfile
    ORDER BY l.logfile ASC;
    """

    for row in execute_many(query, (), log_file):
        yield (row["logfile"], row["duration_us"])


def get_db_log_vehicles(log_file: str) -> Generator[Tuple[str, str], None, None]:
    """
    Get the vehicle used for each log file in the DB, sorted by log file name.
    :param log_file: The log file to query.
    :return: A tuple of (log_name, vehicle_name) for each log file in the database.
    """
    query = """
    SELECT  logfile,
            vehicle_name
    FROM log
    ORDER BY logfile ASC;
    """

    for row in execute_many(query, (), log_file):
        yield (row["logfile"], row["vehicle_name"])


def get_db_scenario_info(log_file: str) -> Generator[Tuple[str, int], None, None]:
    """
    Get the scenario types present in the dictionary and the number of occurances, ordered by occurance count.
    :param log_file: The log file to query.
    :return: A generator of (scenario_tag, count) tuples, ordered by count desc.
    """
    query = """
    SELECT  type,
            COUNT(*) AS cnt
    FROM scenario_tag
    GROUP BY type
    ORDER BY cnt DESC;
    """

    for row in execute_many(query, (), log_file):
        yield (row["type"], row["cnt"])
