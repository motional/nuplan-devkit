from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SensorDataTableRowData:
    """Dummy class to enforce dataclass structure to SensorDataTableRow"""

    pass


class SensorDataTableRow(SensorDataTableRowData, ABC):
    """
    A class representing a row in the SensorData table.
    Each field corresponds to a column in the row.
    """

    @classmethod
    @abstractmethod
    def from_db_row(cls, row: sqlite3.Row) -> SensorDataTableRow:
        """
        A convenience method to convert a row from the SensorData table into a row.
        """
        pass
