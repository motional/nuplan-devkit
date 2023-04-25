from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional, Set

from nuplan.database.nuplan_db.sensor_data_table_row import SensorDataTableRow


@dataclass(frozen=True)
class Camera(SensorDataTableRow):
    """
    A class representing a row in the Image table.
    Each field corresponds to a column in the row.
    """

    token: Optional[str]
    log_token: Optional[str]
    channel: Optional[str]
    model: Optional[str]
    translation: Optional[str]
    rotation: Optional[str]
    intrinsic: Optional[int]
    distortion: Optional[int]
    width: Optional[int]
    height: Optional[int]

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> Camera:
        """
        A convenience method to convert a row from the Camera table into a row.
        :param row: A sqlite row.
        :return: A SensorDataTableRow Image.
        """
        # sqlite3 library doesn't support typing. So ignore for this line.
        keys: Set[str] = set(row.keys())  # type: ignore

        return cls(
            token=row["token"].hex() if "token" in keys else None,
            log_token=row["log_token"].hex() if "log_token" in keys else None,
            channel=row["channel"] if "channel" in keys else None,
            model=row["model"] if "model" in keys else None,
            translation=row["translation"] if "translation" in keys else None,
            rotation=row["rotation"] if "rotation" in keys else None,
            intrinsic=row["intrinsic"] if "intrinsic" in keys else None,
            distortion=row["distortion"] if "distortion" in keys else None,
            width=row["width"] if "width" in keys else None,
            height=row["height"] if "height" in keys else None,
        )
