import os
import time

import typer

from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.database.nuplan_db.scenario_tag import ScenarioTag

cli = typer.Typer()

NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', "/data/sets/nuplan/")
NUPLAN_DB_VERSION = f'{NUPLAN_DATA_ROOT}/nuplan-v1.0/mini/2021.07.16.20.45.29_veh-35_01095_01486.db'


@cli.command()
def info(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out detailed information about the selected database.
    """
    # Construct database
    db = NuPlanDB(load_path=db_version, data_root=data_root)
    # Use the default __str__
    typer.echo("DB info")
    typer.echo(db)


@cli.command()
def duration(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out the duration of the selected db.
    """
    # Construct database
    db = NuPlanDB(load_path=db_version, data_root=data_root)

    # Approximate the duration of db by dividing the number of lidar_pc and the frequency of the DB
    assumed_db_frequency = 20
    db_duration_s = len(db.lidar_pc) / assumed_db_frequency
    db_duration_str = time.strftime("%H:%M:%S", time.gmtime(db_duration_s))
    typer.echo(
        f"DB approximate duration (assuming db frequency {assumed_db_frequency}Hz) is {db_duration_str} [HH:MM:SS]"
    )


@cli.command()
def log_duration(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out the duration of every log in the selected db.
    """
    # Construct database
    db = NuPlanDB(load_path=db_version, data_root=data_root)

    # Approximate the duration of db by dividing the number of lidar_pc and the frequency of the DB
    assumed_db_frequency = 20

    # Print out for every log the approximate durations
    typer.echo(f"The DB: {db.name} contains {len(db.log)} logs")

    for log in db.log:
        lidar_pcs = [lidar for scene in log.scenes for lidar in scene.lidar_pcs]
        db_duration_s = len(lidar_pcs) / assumed_db_frequency
        db_duration_str = time.strftime("%H:%M:%S", time.gmtime(db_duration_s))
        typer.echo(f"\tThe approximate duration of log {log.logfile} is {db_duration_str} [HH:MM:SS]")


@cli.command()
def log_vehicle(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out vehicle information from every log in the selected database.
    """
    # Construct database
    db = NuPlanDB(load_path=db_version, data_root=data_root)

    # Print out for every log the used vehicle
    typer.echo("The used vehicles for every log follow:")

    for log in db.log:
        typer.echo(f"\tFor the log {log.logfile} vehicle {log.vehicle_name} of type {log.vehicle_type} was used")


@cli.command()
def scenarios(
    db_version: str = typer.Argument(NUPLAN_DB_VERSION, help="The database version."),
    data_root: str = typer.Option(NUPLAN_DATA_ROOT, help="The root location of the database"),
) -> None:
    """
    Print out the available scenarios in the selected db.
    """
    # Construct database
    db = NuPlanDB(load_path=db_version, data_root=data_root)

    # Read all available tags:
    available_types = [tag[0] for tag in db.session.query(ScenarioTag.type).distinct().all()]

    # Tag table
    tag_table = db.scenario_tag

    # Print out the available scenarios
    typer.echo(f"The available scenario tags from db: {db_version} follow, in total {len(available_types)} scenarios")

    for tag in available_types:
        tags = tag_table.select_many(type=tag)
        typer.echo(f"\t - {tag} has {len(tags)} scenarios")


if __name__ == '__main__':
    cli()
