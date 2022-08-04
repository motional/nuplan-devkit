from __future__ import annotations  # postpone evaluation of annotations

import logging
from typing import Any

from sqlalchemy import Column, inspect
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import Integer, String

from nuplan.database.common import sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.models import Base

logger = logging.getLogger()


class TrafficLightStatus(Base):
    """
    Traffic Light Statuses in a Log.
    """

    __tablename__ = 'traffic_light_status'

    token: str = Column(sql_types.HexLen8, primary_key=True)
    lidar_pc_token: str = Column(sql_types.HexLen8, ForeignKey("lidar_pc.token"), nullable=False)
    lane_connector_id: int = Column(Integer)
    status: str = Column(String(8))

    lidar_pc: LidarPc = relationship("LidarPc", foreign_keys=[lidar_pc_token], back_populates="traffic_lights")

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session


LidarPc.traffic_lights = relationship(
    "TrafficLightStatus", foreign_keys="TrafficLightStatus.lidar_pc_token", back_populates="lidar_pc"
)
