from __future__ import annotations  # postpone evaluation of annotations

import logging
from typing import Any

from sqlalchemy import Column, inspect
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import Text

from nuplan.database.common import sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.models import Base

logger = logging.getLogger()


class Scene(Base):
    """
    Scenes in a Log.
    """

    __tablename__ = 'scene'

    token: str = Column(sql_types.HexLen8, primary_key=True)
    log_token: str = Column(sql_types.HexLen8, ForeignKey("log.token"), nullable=False)
    name: str = Column(Text)
    goal_ego_pose_token: str = Column(sql_types.HexLen8, ForeignKey("ego_pose.token"), nullable=True)
    roadblock_ids: str = Column(Text)

    goal_ego_pose: EgoPose = relationship("EgoPose", foreign_keys=[goal_ego_pose_token], back_populates="scene")

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


EgoPose.scene = relationship(
    "Scene", foreign_keys="Scene.goal_ego_pose_token", back_populates="goal_ego_pose", uselist=True
)
