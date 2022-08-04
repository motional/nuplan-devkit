from __future__ import annotations  # postpone evaluation of annotations

import logging
import os
import os.path as osp
from typing import TYPE_CHECKING, Any, BinaryIO, List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sqlalchemy import Column, inspect
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import Integer, String

from nuplan.database.common import sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.models import Base
from nuplan.database.nuplan_db_orm.scene import Scene
from nuplan.database.nuplan_db_orm.utils import get_boxes, get_future_box_sequence, pack_future_boxes, render_on_map
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.label.label import Label
from nuplan.database.utils.label.utils import raw_mapping
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

logger = logging.getLogger()

if TYPE_CHECKING:
    from nuplan.database.nuplan_db_orm.log import Log
    from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB


class LidarPc(Base):
    """
    A lidar point cloud.
    """

    __tablename__ = "lidar_pc"

    token = Column(sql_types.HexLen8, primary_key=True)  # type: str
    next_token = Column(sql_types.HexLen8, ForeignKey("lidar_pc.token"), nullable=True)  # type: str
    prev_token = Column(sql_types.HexLen8, ForeignKey("lidar_pc.token"), nullable=True)  # type: str
    ego_pose_token = Column(sql_types.HexLen8, ForeignKey("ego_pose.token"), nullable=False)  # type: str
    lidar_token = Column(sql_types.HexLen8, ForeignKey("lidar.token"), nullable=False)  # type: str
    scene_token = Column(sql_types.HexLen8, ForeignKey("scene.token"), nullable=False)  # type: str
    filename = Column(String(128))  # type: str
    timestamp = Column(Integer)  # field type: int

    next = relationship("LidarPc", foreign_keys=[next_token], remote_side=[token])  # type: LidarPc
    prev = relationship("LidarPc", foreign_keys=[prev_token], remote_side=[token])  # type: LidarPc
    ego_pose = relationship("EgoPose", foreign_keys=[ego_pose_token], back_populates="lidar_pc")  # type: EgoPose
    scene = relationship("Scene", foreign_keys=[scene_token], back_populates="lidar_pcs")  # type: Scene
    lidar_boxes = relationship("LidarBox", foreign_keys="LidarBox.lidar_pc_token", back_populates="lidar_pc")

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
    def log(self) -> Log:
        """
        Returns the Log containing the LidarPC.
        :return: The log containing the LidarPC.
        """
        return self.lidar.log

    def future_ego_pose(self) -> Optional[EgoPose]:
        """
        Get future ego poses.
        :return: Ego pose at next pointcloud if any.
        """
        if self.next is not None:
            return self.next.ego_pose
        return None

    def past_ego_pose(self) -> Optional[EgoPose]:
        """
        Get past ego poses.
        :return: Ego pose at previous pointcloud if any.
        """
        if self.prev is not None:
            return self.prev.ego_pose
        return None

    def future_or_past_ego_poses(self, number: int, mode: str, direction: str) -> List[EgoPose]:
        """
        Get n future or past vehicle poses. Note here the frequency of pose differs from frequency of LidarPc.
        :param number: Number of poses to fetch or number of seconds of ego poses to fetch.
        :param mode: Either n_poses or n_seconds.
        :param direction: Future or past ego poses to fetch, could be 'prev' or 'next'.
        :return: List of up to n or n seconds future or past ego poses.
        """
        if direction == 'prev':
            if mode == 'n_poses':
                return (  # type: ignore
                    self._session.query(EgoPose)
                    .filter(EgoPose.timestamp < self.ego_pose.timestamp, self.lidar.log_token == EgoPose.log_token)
                    .order_by(EgoPose.timestamp.desc())
                    .limit(number)
                    .all()
                )
            elif mode == 'n_seconds':
                return (  # type: ignore
                    self._session.query(EgoPose)
                    .filter(
                        EgoPose.timestamp - self.ego_pose.timestamp < 0,
                        EgoPose.timestamp - self.ego_pose.timestamp >= -number * 1e6,
                        self.lidar.log_token == EgoPose.log_token,
                    )
                    .order_by(EgoPose.timestamp.desc())
                    .all()
                )
            else:
                raise ValueError(f"Unknown mode: {mode}.")
        elif direction == 'next':
            if mode == 'n_poses':
                return (  # type: ignore
                    self._session.query(EgoPose)
                    .filter(EgoPose.timestamp > self.ego_pose.timestamp, self.lidar.log_token == EgoPose.log_token)
                    .order_by(EgoPose.timestamp.asc())
                    .limit(number)
                    .all()
                )
            elif mode == 'n_seconds':
                return (  # type: ignore
                    self._session.query(EgoPose)
                    .filter(
                        EgoPose.timestamp - self.ego_pose.timestamp > 0,
                        EgoPose.timestamp - self.ego_pose.timestamp <= number * 1e6,
                        self.lidar.log_token == EgoPose.log_token,
                    )
                    .order_by(EgoPose.timestamp.asc())
                    .all()
                )
            else:
                raise ValueError(f"Unknown mode: {mode}.")
        else:
            raise ValueError(f"Unknown direction: {direction}.")

    def load(self, db: NuPlanDB, remove_close: bool = True) -> LidarPointCloud:
        """
        Load a point cloud.
        :param db: Log Database.
        :param remove_close: If true, remove nearby points, defaults to True.
        :return: Loaded point cloud.
        """
        if self.lidar.channel == 'MergedPointCloud':
            if self.filename.endswith('bin2'):
                return LidarPointCloud.from_buffer(self.load_bytes(db), 'bin2')
            else:
                # load pcd file
                assert self.filename.endswith('pcd'), f'.pcd file is expected but get {self.filename}'
                return LidarPointCloud.from_buffer(self.load_bytes(db), 'pcd')
        else:
            raise NotImplementedError

    def load_bytes(self, db: NuPlanDB) -> BinaryIO:
        """
        Load the point cloud in binary.
        :param db: Log Database.
        :return: Point cloud bytes.
        """
        blob: BinaryIO = db.load_blob(os.path.join("sensor_blobs", self.filename))
        return blob

    def path(self, db: NuPlanDB) -> str:
        """
        Get the path to the point cloud file.
        :param db: Log Database.
        :return: Point cloud file path.
        """
        self.load_bytes(db)
        return osp.join(db.data_root, self.filename)

    def boxes(self, frame: Frame = Frame.GLOBAL) -> List[Box3D]:
        """
        Loads all boxes associated with this LidarPc record. Boxes are returned in the global frame by default.
        :param frame: Specify the frame in which the boxes will be returned.
        :return: The list of boxes.
        """
        boxes: List[Box3D] = get_boxes(self, frame, self.ego_pose.trans_matrix_inv, self.lidar.trans_matrix_inv)

        return boxes

    def boxes_with_future_waypoints(
        self, future_horizon_len_s: float, future_interval_s: float, frame: Frame = Frame.GLOBAL
    ) -> List[Box3D]:
        """
        Loads all boxes and future boxes associated with this LidarPc record. Boxes are returned in the global frame by
            default and annotations are sampled at a frequency of ~0.5 seconds.
        :param future_horizon_len_s: Timestep horizon of the future waypoints in seconds.
        :param future_interval_s: Timestep interval of the future waypoints in seconds.
        :param frame: Specify the frame in which the boxes will be returned.
        :return: List of boxes in sample data that includes box centers and orientations at future timesteps.
        """
        # Because the 6 sec sample could have a timestamp that is slightly larger than 6 sec (e.g., 6.0001 sec),
        # we need to read more samples to make sure the sequence includes all the timestamps in the horizon.
        TIMESTAMP_MARGIN_MS = 1e6
        future_horizon_len_ms = future_horizon_len_s * 1e6
        query = (
            self._session.query(LidarPc)
            .filter(
                LidarPc.timestamp - self.timestamp >= 0,
                LidarPc.timestamp - self.timestamp <= future_horizon_len_ms + TIMESTAMP_MARGIN_MS,
            )
            .order_by(LidarPc.timestamp.asc())
            .all()
        )
        lidar_pcs = [lidar_pc for lidar_pc in list(query)]

        track_token_2_box_sequence = get_future_box_sequence(
            lidar_pcs=lidar_pcs,
            frame=frame,
            future_horizon_len_s=future_horizon_len_s,
            future_interval_s=future_interval_s,
            trans_matrix_ego=self.ego_pose.trans_matrix_inv,
            trans_matrix_sensor=self.lidar.trans_matrix_inv,
        )
        boxes_with_future_waypoints: List[Box3D] = pack_future_boxes(
            track_token_2_box_sequence=track_token_2_box_sequence,
            future_interval_s=future_interval_s,
            future_horizon_len_s=future_horizon_len_s,
        )

        return boxes_with_future_waypoints

    def render(
        self,
        db: NuPlanDB,
        render_future_waypoints: bool = False,
        render_map_raster: bool = False,
        render_vector_map: bool = False,
        render_track_color: bool = False,
        render_future_ego_poses: bool = False,
        track_token: Optional[str] = None,
        with_anns: bool = True,
        axes_limit: float = 80.0,
        ax: Axes = None,
    ) -> plt.axes:
        """
        Render the Lidar pointcloud with appropriate boxes and (optionally) the map raster.
        :param db: Log database.
        :param render_future_waypoints: Whether to render future waypoints.
        :param render_map_raster: Whether to render the map raster.
        :param render_vector_map: Whether to render the vector map.
        :param render_track_color: Whether to render the tracks with different random color.
        :param render_future_ego_poses: Whether to render future ego poses.
        :param track_token: Which instance to render, if it's None, render all the instances.
        :param with_anns: Whether you want to render the annotations?
        :param axes_limit: The range of Lidar pointcloud that will be rendered will be between
            (-axes_limit, axes_limit).
        :param ax: Axes object.
        :return: Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(25, 25))

        if with_anns:
            if render_future_waypoints:
                DEFAULT_FUTURE_HORIZON_LEN_S = 6.0
                DEFAULT_FUTURE_INTERVAL_S = 0.5
                boxes = self.boxes_with_future_waypoints(
                    DEFAULT_FUTURE_HORIZON_LEN_S, DEFAULT_FUTURE_INTERVAL_S, Frame.SENSOR
                )
            else:
                boxes = self.boxes(Frame.SENSOR)
        else:
            boxes = []

        if render_future_ego_poses:
            DEFAULT_FUTURE_HORIZON_LEN_S = 6
            TIMESTAMP_MARGIN_S = 1
            ego_poses = self.future_or_past_ego_poses(
                DEFAULT_FUTURE_HORIZON_LEN_S + TIMESTAMP_MARGIN_S, 'n_seconds', 'next'
            )
        else:
            ego_poses = [self.ego_pose]

        labelmap = {
            lid: Label(raw_mapping['id2local'][lid], raw_mapping['id2color'][lid])
            for lid in raw_mapping['id2local'].keys()
        }

        render_on_map(
            lidarpc_rec=self,
            db=db,
            boxes_lidar=boxes,
            ego_poses=ego_poses,
            radius=axes_limit,
            ax=ax,
            labelmap=labelmap,
            render_map_raster=render_map_raster,
            render_vector_map=render_vector_map,
            track_token=track_token,
            with_random_color=render_track_color,
            render_future_ego_poses=render_future_ego_poses,
        )

        plt.axis('equal')
        ax.set_title('PC {} from {} in {}'.format(self.token, self.lidar.channel, self.log.location))

        return ax


EgoPose.lidar_pc = relationship(
    "LidarPc", foreign_keys="LidarPc.ego_pose_token", back_populates="ego_pose", uselist=False
)
Scene.lidar_pcs = relationship("LidarPc", foreign_keys=[LidarPc.scene_token], back_populates="scene")
LidarBox.lidar_pc = relationship("LidarPc", foreign_keys=[LidarBox.lidar_pc_token], back_populates="lidar_boxes")
