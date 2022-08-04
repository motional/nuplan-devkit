from __future__ import annotations  # postpone evaluation of annotations

import os.path as osp
from typing import TYPE_CHECKING, Any, BinaryIO, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib.axes import Axes
from sqlalchemy import Column, func, inspect
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import Integer, String

from nuplan.database.common import sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.models import Base
from nuplan.database.nuplan_db_orm.scene import Scene
from nuplan.database.nuplan_db_orm.utils import get_boxes
from nuplan.database.utils.boxes.box3d import Box3D, BoxVisibility, box_in_image

if TYPE_CHECKING:
    from nuplan.database.common.db import NuPlanDB
    from nuplan.database.nuplan_db_orm.log import Log


class Image(Base):
    """
    An image.
    """

    __tablename__ = "image"

    token = Column(sql_types.HexLen8, primary_key=True)  # type: str
    next_token = Column(sql_types.HexLen8, ForeignKey("image.token"), nullable=True)  # type: str
    prev_token = Column(sql_types.HexLen8, ForeignKey("image.token"), nullable=True)  # type: str
    ego_pose_token = Column(sql_types.HexLen8, ForeignKey("ego_pose.token"), nullable=False)  # type: str
    camera_token = Column(sql_types.HexLen8, ForeignKey("camera.token"), nullable=False)  # type: str
    filename_jpg = Column(String(128))  # type: str
    timestamp = Column(Integer)  # type: int

    next = relationship("Image", foreign_keys=[next_token], remote_side=[token])  # type: Image
    prev = relationship("Image", foreign_keys=[prev_token], remote_side=[token])  # type: Image
    camera = relationship("Camera", foreign_keys=[camera_token], back_populates="images")  # type: Camera
    ego_pose = relationship("EgoPose", foreign_keys=[ego_pose_token], back_populates="image")  # type: EgoPose

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
    def log(self) -> Log:
        """
        Returns the Log containing the image.
        :return: The log containing this image.
        """
        return self.camera.log

    @property
    def lidar_pc(self) -> LidarPc:
        """
        Get the closest LidarPc by timestamp
        :return: LidarPc closest to the Image by time
        """
        lidar_pc = self._session.query(LidarPc).order_by(func.abs(LidarPc.timestamp - self.timestamp)).first()
        return lidar_pc

    @property
    def scene(self) -> Scene:
        """
        Get the corresponding scene by finding the closest LidarPc by timestamp.
        :return: Scene corresponding to the Image.
        """
        return self.lidar_pc.scene

    @property
    def lidar_boxes(self) -> LidarBox:
        """
        Get the list of boxes associated with this Image, based on closest LidarPc
        :return: List of boxes associated with this Image
        """
        return self.lidar_pc.lidar_boxes

    def load_as(self, db: NuPlanDB, img_type: str) -> Any:
        """
        Loads the image as a desired type.
        :param db: Log Database.
        :param img_type: Can be either 'pil' or 'np' or 'cv2'. If the img_type is cv2, the image is returned in BGR
            format, otherwise it is returned in RGB format.
        :return: The image.
        """
        assert img_type in ["pil", "cv2", "np"], f"Expected img_type to be pil, cv2 or np. Received {img_type}"

        pil_img = PIL.Image.open(self.load_bytes_jpg(db))

        if img_type == "pil":
            return pil_img
        elif img_type == "np":
            return np.array(pil_img)
        else:
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @property
    def filename(self) -> str:
        """
        Get the file name.
        :return: The file name.
        """
        return self.filename_jpg

    def load_bytes_jpg(self, db: NuPlanDB) -> BinaryIO:
        """
        Returns the bytes of the jpg data for this image.
        :param db: Log Database.
        :return: The image bytes.
        """
        blob: BinaryIO = db.load_blob(osp.join("sensor_blobs", self.filename))
        return blob

    def path(self, db: NuPlanDB) -> str:
        """
        Get the path to image file.
        :param db: Log Database.
        :return: The image file path.
        """
        return osp.join(db.data_root, self.filename)

    def boxes(self, frame: Frame = Frame.GLOBAL) -> List[Box3D]:
        """
        Loads all boxes associated with this Image record. Boxes are returned in the global frame by default.
        :param frame: Specify the frame in which the boxes will be returned.
        :return: List of boxes.
        """
        boxes: List[Box3D] = get_boxes(self, frame, self.ego_pose.trans_matrix_inv, self.camera.trans_matrix_inv)

        return boxes

    def future_or_past_ego_poses(self, number: int, mode: str, direction: str) -> List[EgoPose]:
        """
        Get n future or past vehicle poses. Note here the frequency of pose differs from frequency of Image.
        :param number: Number of poses to fetch or number of seconds of ego poses to fetch.
        :param mode: Either n_poses or n_seconds.
        :param direction: Future or past ego poses to fetch, could be 'prev' or 'next'.
        :return: List of up to n or n seconds future or past ego poses.
        """
        ego_poses: List[EgoPose]

        if direction == "prev":
            if mode == "n_poses":
                ego_poses = (
                    self._session.query(EgoPose)
                    .filter(EgoPose.timestamp < self.ego_pose.timestamp, self.camera.log_token == EgoPose.log_token)
                    .order_by(EgoPose.timestamp.desc())
                    .limit(number)
                    .all()
                )
                return ego_poses
            elif mode == "n_seconds":
                ego_poses = (
                    self._session.query(EgoPose)
                    .filter(
                        EgoPose.timestamp - self.ego_pose.timestamp < 0,
                        EgoPose.timestamp - self.ego_pose.timestamp >= -number * 1e6,
                        self.camera.log_token == EgoPose.log_token,
                    )
                    .order_by(EgoPose.timestamp.desc())
                    .all()
                )
                return ego_poses
            else:
                raise NotImplementedError("Only n_poses and n_seconds two modes are supported for now!")
        elif direction == "next":
            if mode == "n_poses":
                ego_poses = (
                    self._session.query(EgoPose)
                    .filter(EgoPose.timestamp > self.ego_pose.timestamp, self.camera.log_token == EgoPose.log_token)
                    .order_by(EgoPose.timestamp.asc())
                    .limit(number)
                    .all()
                )
                return ego_poses
            elif mode == "n_seconds":
                ego_poses = (
                    self._session.query(EgoPose)
                    .filter(
                        EgoPose.timestamp - self.ego_pose.timestamp > 0,
                        EgoPose.timestamp - self.ego_pose.timestamp <= number * 1e6,
                        self.camera.log_token == EgoPose.log_token,
                    )
                    .order_by(EgoPose.timestamp.asc())
                    .all()
                )
                return ego_poses
            else:
                raise NotImplementedError("Only n_poses and n_seconds two modes are supported!")
        else:
            raise ValueError("Only prev and next two directions are supported!")

    def render(
        self,
        db: NuPlanDB,
        with_3d_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        ax: Optional[Axes] = None,
    ) -> None:
        """
        Render the image with all 3d and 2d annotations.
        :param db: Log Database.
        :param with_3d_anns: Whether you want to render 3D boxes?
        :param box_vis_level: One of the enumerations of <BoxVisibility>.
        :param ax: Axes object or array of Axes objects.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 16))

        ax.imshow(self.load_as(db, img_type="pil"))

        if with_3d_anns:
            for box in self.boxes(Frame.SENSOR):

                # Get the LidarBox record with the same token as box.token
                ann_record = db.lidar_box[box.token]

                c = ann_record.category.color_np
                color = c, c, np.array([0, 0, 0])  # type: Tuple[Any, Any, np.typing.NDArray[np.int64]]

                if box_in_image(
                    box, self.camera.intrinsic_np, (self.camera.width, self.camera.height), vis_level=box_vis_level
                ):
                    box.render(ax, view=self.camera.intrinsic_np, normalize=True, colors=color)

        ax.set_xlim(0, self.camera.width)
        ax.set_ylim(self.camera.height, 0)
        ax.set_title(self.camera.channel)


Camera.images = relationship("Image", foreign_keys="Image.camera_token", back_populates="camera")
EgoPose.image = relationship("Image", foreign_keys="Image.ego_pose_token", back_populates="ego_pose", uselist=False)
