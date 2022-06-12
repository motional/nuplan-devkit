#!/usr/bin/env python

import typer

from nuplan.cli import db_cli

# Construct main cli interface
cli = typer.Typer()

# Add database CLI
cli.add_typer(db_cli.cli, name="db")


def main() -> None:
    """
    Main entry point for the CLI
    """
    cli()


if __name__ == '__main__':
    main()
