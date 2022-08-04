from __future__ import annotations  # postpone evaluation of annotations

import logging
from typing import Any

from sqlalchemy import Column, inspect
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import Text

from nuplan.database.common import sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.models import Base

logger = logging.getLogger()


class ScenarioTag(Base):
    """
    Scenarios Tags for a scene.
    """

    __tablename__ = 'scenario_tag'

    token: str = Column(sql_types.HexLen8, primary_key=True)
    lidar_pc_token: str = Column(sql_types.HexLen8, ForeignKey("lidar_pc.token"), nullable=False)
    type: str = Column(Text)
    agent_track_token: str = Column(sql_types.HexLen8, ForeignKey("track.token"), nullable=False)

    lidar_pc: LidarPc = relationship("LidarPc", foreign_keys=[lidar_pc_token], back_populates="scenario_tags")

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc


LidarPc.scenario_tags = relationship(
    "ScenarioTag", foreign_keys="ScenarioTag.lidar_pc_token", back_populates="lidar_pc"
)
