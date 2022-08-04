from __future__ import annotations  # postpone evaluation of annotations

import logging
from typing import Any, List

from sqlalchemy import Column, inspect
from sqlalchemy.orm import relationship
from sqlalchemy.types import Integer, String

from nuplan.database.common import sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.image import Image
from nuplan.database.nuplan_db_orm.lidar import Lidar
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.models import Base
from nuplan.database.nuplan_db_orm.scene import Scene

logger = logging.getLogger()


class Log(Base):
    """
    Information about the log from which the data was extracted.
    """

    __tablename__ = "log"

    token = Column(sql_types.HexLen8, primary_key=True)  # type: str
    vehicle_name = Column(String(64))  # type: str
    date = Column(String(64))  # type: str
    timestamp = Column(Integer)  # type: int
    logfile = Column(String(64))  # type: str
    location = Column(String(64))  # type: str
    map_version = Column(String(64))  # type: str

    cameras = relationship("Camera", foreign_keys="Camera.log_token", back_populates="log")  # type: List[Camera]
    ego_poses = relationship("EgoPose", foreign_keys="EgoPose.log_token", back_populates="log")  # type: List[EgoPose]
    lidars = relationship("Lidar", foreign_keys="Lidar.log_token", back_populates="log")  # type: List[Lidar]
    scenes = relationship("Scene", foreign_keys="Scene.log_token", back_populates="log")  # type: List[Scene]

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    @property
    def images(self) -> List[Image]:
        """
        Returns list of Images contained in the Log.
        :return: The list of Images contained in the log.
        """
        log_images = []
        for camera in self.cameras:
            log_images.extend(camera.images)
        return log_images

    @property
    def lidar_pcs(self) -> List[LidarPc]:
        """
        Returns list of Lidar PCs in the Log.
        :return: The list of Lidar PCs in the log.
        """
        log_lidar_pcs = []
        for lidar in self.lidars:
            log_lidar_pcs.extend(lidar.lidar_pcs)
        return log_lidar_pcs

    @property
    def lidar_boxes(self) -> List[LidarBox]:
        """
        Returns list of Lidar Boxes in the Log.
        :return: The list of Lidar Boxes in the log.
        """
        log_lidar_boxes = []
        for lidar_pc in self.lidar_pcs:
            log_lidar_boxes.extend(lidar_pc.lidar_boxes)
        return log_lidar_boxes

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc


Camera.log = relationship("Log", foreign_keys=[Camera.log_token], back_populates="cameras")
EgoPose.log = relationship("Log", foreign_keys=[EgoPose.log_token], back_populates="ego_poses")
Lidar.log = relationship("Log", foreign_keys=[Lidar.log_token], back_populates="lidars")
Scene.log = relationship("Log", foreign_keys=[Scene.log_token], back_populates="scenes")
