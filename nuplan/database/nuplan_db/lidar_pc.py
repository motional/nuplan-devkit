from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional, Set

from nuplan.database.nuplan_db.sensor_data_table_row import SensorDataTableRow


@dataclass(frozen=True)
class LidarPc(SensorDataTableRow):
    """
    A class representing a row in the LidarPC table.
    Each field corresponds to a column in the row.
    """

    token: Optional[str]
    next_token: Optional[str]
    prev_token: Optional[str]
    ego_pose_token: Optional[str]
    lidar_token: Optional[str]
    scene_token: Optional[str]
    filename: Optional[str]
    timestamp: Optional[int]

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> LidarPc:
        """
        A convenience method to convert a row from the LidarPc table into a row.
        """
        # sqlite3 library doesn't support typing. So ignore for this line.
        keys: Set[str] = set(row.keys())  # type: ignore

        return cls(
            token=row["token"].hex() if "token" in keys else None,
            next_token=row["next_token"].hex() if "next_token" in keys else None,
            prev_token=row["prev_token"].hex() if "prev_token" in keys else None,
            ego_pose_token=row["ego_pose_token"].hex() if "ego_pose_token" in keys else None,
            lidar_token=row["lidar_token"].hex() if "lidar_token" in keys else None,
            scene_token=row["scene_token"].hex() if "scene_token" in keys else None,
            filename=row["filename"] if "filename" in keys else None,
            timestamp=row["timestamp"] if "timestamp" in keys else None,
        )
