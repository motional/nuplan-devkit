from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional, Set

from nuplan.database.nuplan_db.sensor_data_table_row import SensorDataTableRow


@dataclass(frozen=True)
class Image(SensorDataTableRow):
    """
    A class representing a row in the Image table.
    Each field corresponds to a column in the row.
    """

    token: Optional[str]
    next_token: Optional[str]
    prev_token: Optional[str]
    ego_pose_token: Optional[str]
    camera_token: Optional[str]
    filename_jpg: Optional[str]
    timestamp: Optional[int]
    channel: Optional[str]

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> Image:
        """
        A convenience method to convert a row from the Image table into a row.
        :param row: A sqlite row.
        :return: A SensorDataTableRow Image.
        """
        # sqlite3 library doesn't support typing. So ignore for this line.
        keys: Set[str] = set(row.keys())  # type: ignore

        return cls(
            token=row["token"].hex() if "token" in keys else None,
            next_token=row["next_token"].hex() if "next_token" in keys and row["next_token"] is not None else None,
            prev_token=row["prev_token"].hex() if "prev_token" in keys and row["prev_token"] is not None else None,
            ego_pose_token=row["ego_pose_token"].hex() if "ego_pose_token" in keys else None,
            camera_token=row["camera_token"].hex() if "camera_token" in keys else None,
            filename_jpg=row["filename_jpg"] if "filename_jpg" in keys else None,
            timestamp=row["timestamp"] if "timestamp" in keys else None,
            channel=row["channel"] if "channel" in keys else None,
        )
