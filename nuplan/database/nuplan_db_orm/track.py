from __future__ import annotations  # postpone evaluation of annotations

from typing import Any, List

import numpy as np
import numpy.typing as npt
from sqlalchemy import Column, inspect
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import Float

from nuplan.database.common import sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.category import Category
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.models import Base
from nuplan.database.nuplan_db_orm.scenario_tag import ScenarioTag


class Track(Base):
    """
    Track from tracker output. A track represents a bunch of lidar boxes with the same instance id in a given log.
    """

    __tablename__ = "track"

    token: str = Column(sql_types.HexLen8, primary_key=True)
    category_token: str = Column(sql_types.HexLen8, ForeignKey("category.token"), nullable=False)
    width: float = Column(Float)
    length: float = Column(Float)
    height: float = Column(Float)

    lidar_boxes: List[LidarBox] = relationship("LidarBox", foreign_keys=[LidarBox.track_token], back_populates="track")
    scenario_tags: List[ScenarioTag] = relationship(
        "ScenarioTag", foreign_keys=[ScenarioTag.agent_track_token], back_populates="agent_track"
    )
    category: Category = relationship("Category", foreign_keys=[category_token], back_populates="tracks")

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

    @property
    def nbr_lidar_boxes(self) -> int:
        """
        Returns number of boxes in the Track.
        :return: Number of boxes.
        """
        nbr: int = self._session.query(LidarBox).filter(LidarBox.track_token == self.token).count()
        return nbr

    @property
    def first_lidar_box(self) -> LidarBox:
        """
        Returns first lidar box along the track.
        :return: First lidar box along the track.
        """
        box: LidarBox = (
            self._session.query(LidarBox)
            .filter(LidarBox.track_token == self.token)
            .join(LidarPc)
            .order_by(LidarPc.timestamp.asc())
            .first()
        )
        return box

    @property
    def last_lidar_box(self) -> LidarBox:
        """
        Returns last lidar box along the track.
        :return: Last lidar box along the track.
        """
        box: LidarBox = (
            self._session.query(LidarBox)
            .filter(LidarBox.track_token == self.token)
            .join(LidarPc)
            .order_by(LidarPc.timestamp.desc())
            .first()
        )
        return box

    @property
    def duration(self) -> int:
        """
        Returns duration of Track.
        :return: Duration of the track.
        """
        d: int = self.last_lidar_box.timestamp - self.first_lidar_box.timestamp
        return d

    @property
    def distances_to_ego(self) -> npt.NDArray[np.float64]:
        """
        Returns array containing distances of all boxes in the Track from ego vehicle.
        :return: Distances of all boxes in the track from ego vehicle.
        """
        return np.asarray([lidar_box.distance_to_ego for lidar_box in self.lidar_boxes])

    @property
    def min_distance_to_ego(self) -> float:
        """
        Returns minimum distance of Track from Ego Vehicle.
        :return: The minimum distance of the track from ego vehicle.
        """
        min_dist: float = np.amin(self.distances_to_ego)
        return min_dist

    @property
    def max_distance_to_ego(self) -> float:
        """
        Returns maximum distance of Track from Ego Vehicle.
        :return: The maximum distance of the tack from ego vehicle.
        """
        max_dist: float = np.amax(self.distances_to_ego)
        return max_dist


LidarBox.track = relationship("Track", foreign_keys=[LidarBox.track_token], back_populates="lidar_boxes")
ScenarioTag.agent_track = relationship(
    "Track", foreign_keys=[ScenarioTag.agent_track_token], back_populates="scenario_tags"
)
Category.tracks = relationship("Track", foreign_keys="Track.category_token", back_populates="category")
