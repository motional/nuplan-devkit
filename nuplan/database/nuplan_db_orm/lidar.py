from __future__ import annotations  # postpone evaluation of annotations

import logging
from typing import Any, List

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion
from sqlalchemy import Column, inspect
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import String

from nuplan.database.common import data_types, sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.models import Base

logger = logging.getLogger()


class Lidar(Base):
    """
    Defines a calibrated lidar used to record a particular log.
    """

    __tablename__ = "lidar"

    token = Column(sql_types.HexLen8, primary_key=True)  # type: str
    log_token = Column(sql_types.HexLen8, ForeignKey("log.token"), nullable=False)  # type: str
    channel = Column(String(64))  # type: str
    model = Column(String(64))  # type: str
    translation = Column(sql_types.SqlTranslation)  # type: data_types.Translation
    rotation = Column(sql_types.SqlRotation)  # type: data_types.Rotation

    lidar_pcs = relationship(
        "LidarPc", foreign_keys="LidarPc.lidar_token", back_populates="lidar"
    )  # type: List[LidarPc]

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def translation_np(self) -> npt.NDArray[np.float64]:
        """
        Get the translation in numpy format.
        :return: <np.float: 3> Translation.
        """
        return np.array(self.translation)

    @property
    def quaternion(self) -> Quaternion:
        """
        Get the rotation in quaternion.
        :return: The rotation in quaternion.
        """
        return Quaternion(self.rotation)

    @property
    def trans_matrix(self) -> npt.NDArray[np.float64]:
        """
        Get the transformation matrix.
        :return: <np.float: 4, 4>. Transformation matrix.
        """
        tm: npt.NDArray[np.float64] = self.quaternion.transformation_matrix
        tm[:3, 3] = self.translation_np
        return tm

    @property
    def trans_matrix_inv(self) -> npt.NDArray[np.float64]:
        """
        Get the inverse transformation matrix.
        :return: <np.float: 4, 4>. Inverse transformation matrix.
        """
        tm: npt.NDArray[np.float64] = np.eye(4)
        rot_inv = self.quaternion.rotation_matrix.T
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(np.transpose(-self.translation_np))
        return tm


LidarPc.lidar = relationship("Lidar", foreign_keys=[LidarPc.lidar_token], back_populates="lidar_pcs")
