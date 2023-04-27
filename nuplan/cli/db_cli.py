import os
import time

import typer

from nuplan.database.nuplan_db.db_cli_queries import (
    get_db_description,
    get_db_duration_in_us,
    get_db_log_duration,
    get_db_log_vehicles,
    get_db_scenario_info,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import download_file_if_necessary

cli = typer.Typer()

NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', "/data/sets/nuplan/")
NUPLAN_DB_VERSION = f'{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.07.16.20.45.29_veh-35_01095_01486.db'


def _ensure_file_downloaded(data_root: str, potentially_remote_path: str) -> str:
    """
    Attempts to download the DB file from a remote URL if it does not exist locally.
    If the download fails, an error will be raised.
    :param data_root: The location to download the file, if necessary.
    :param potentially_remote_path: The path to the file.
    :return: The resulting file path. Will be one of a few options:
        * If potentially_remote_path points to a local file, will return potentially_remote_path
        * If potentially_remote_file points to a remote file, it does not exist currently, and the file can be successfully downloaded, it will return the path of the downloaded file.
        * In all other cases, an error will be raised.
    """
    output_file_path: str = download_file_if_necessary(data_root, potentially_remote_path)

    if not os.path.exists(output_file_path):
        raise ValueError(f"{potentially_remote_path} could not be downloaded.")

    return output_file_path


@cli.command()
def info(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out detailed information about the selected database.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    db_description = get_db_description(db_version)

    for table_name, table_description in db_description.tables.items():
        typer.echo(f"Table {table_name}: {table_description.row_count} rows")

        for column_name, column_description in table_description.columns.items():
            typer.echo(
                "".join(
                    [
                        f"\tcolumn {column_name}: {column_description.data_type} ",
                        "NULL " if column_description.nullable else "NOT NULL ",
                        "PRIMARY KEY " if column_description.is_primary_key else "",
                    ]
                )
            )

        typer.echo()


@cli.command()
def duration(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out the duration of the selected db.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    db_duration_us = get_db_duration_in_us(db_version)
    db_duration_s = float(db_duration_us) / 1e6
    db_duration_str = time.strftime("%H:%M:%S", time.gmtime(db_duration_s))
    typer.echo(f"DB duration is {db_duration_str} [HH:MM:SS]")


@cli.command()
def log_duration(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out the duration of every log in the selected db.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    num_logs = 0
    for log_file_name, log_file_duration_us in get_db_log_duration(db_version):
        log_file_duration_s = float(log_file_duration_us) / 1e6
        log_file_duration_str = time.strftime("%H:%M:%S", time.gmtime(log_file_duration_s))
        typer.echo(f"The duration of log {log_file_name} is {log_file_duration_str} [HH:MM:SS]")
        num_logs += 1

    typer.echo(f"There are {num_logs} total logs.")


@cli.command()
def log_vehicle(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out vehicle information from every log in the selected database.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    for log_file, vehicle_name in get_db_log_vehicles(db_version):
        typer.echo(f"For the log {log_file}, vehicle {vehicle_name} was used.")


@cli.command()
def scenarios(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out the available scenarios in the selected db.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    total_count = 0
    for tag, num_scenarios in get_db_scenario_info(db_version):
        typer.echo(f"{tag}: {num_scenarios} scenarios.")
        total_count += num_scenarios

    typer.echo(f"TOTAL: {total_count} scenarios.")


if __name__ == '__main__':
    cli()
