from __future__ import annotations  # postpone evaluation of annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion
from sqlalchemy import Column, inspect
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import Integer, PickleType, String

from nuplan.database.common import data_types, sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.models import Base


class Camera(Base):
    """
    Defines a calibrated camera used to record a particular log.
    """

    __tablename__ = "camera"

    token = Column(sql_types.HexLen8, primary_key=True)  # type: str
    log_token = Column(sql_types.HexLen8, ForeignKey("log.token"), nullable=False)  # type: str
    channel = Column(String(64))  # type: str
    model = Column(String(64))  # type: str
    translation = Column(sql_types.SqlTranslation)  # type: data_types.Translation
    rotation = Column(sql_types.SqlRotation)  # type: data_types.Rotation
    intrinsic = Column(sql_types.SqlCameraIntrinsic)  # type: data_types.CameraIntrinsic
    distortion = Column(PickleType)  # type: list[float]
    width = Column(Integer)  # type: int
    height = Column(Integer)  # type: int

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
        :return : The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def intrinsic_np(self) -> npt.NDArray[np.float64]:
        """
        Get the intrinsic in numpy format.
        :return: <np.float: 3, 3> Camera intrinsic.
        """
        return np.array(self.intrinsic)

    @property
    def distortion_np(self) -> npt.NDArray[np.float64]:
        """
        Get the distortion in numpy format.
        :return: <np.float: N> Camera distrotion.
        """
        return np.array(self.distortion)

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
        :return: Rotation in quaternion.
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
