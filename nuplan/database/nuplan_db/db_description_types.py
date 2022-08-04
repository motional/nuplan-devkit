from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ColumnDescription:
    """
    A description of a column present in a Sqlite DB
    """

    column_id: int
    name: int
    data_type: str
    nullable: bool
    is_primary_key: bool


@dataclass(frozen=True)
class TableDescription:
    """
    A description of a table present in a Sqlite DB
    """

    # A mapping of [name, ColumnDescription]
    columns: Dict[str, ColumnDescription]
    row_count: int
    name: str


@dataclass(frozen=True)
class DbDescription:
    """
    A description of a Sqlite DB.
    """

    # A mapping of [name, TableColumn]
    tables: Dict[str, TableDescription]
