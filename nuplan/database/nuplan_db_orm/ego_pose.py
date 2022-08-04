from __future__ import annotations  # postpone evaluation of annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from sqlalchemy import Column, inspect
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import Float, Integer

from nuplan.database.common import sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.maps_db.utils import build_lane_segments_from_blps, connect_blp_predecessor, connect_blp_successor
from nuplan.database.nuplan_db_orm.models import Base
from nuplan.database.nuplan_db_orm.utils import crop_rect, generate_multi_scale_connections, get_candidates
from nuplan.database.nuplan_db_orm.vector_map_np import VectorMapNp

logger = logging.getLogger()


class EgoPose(Base):
    """
    Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system.
    """

    __tablename__ = "ego_pose"

    token = Column(sql_types.HexLen8, primary_key=True)  # type: str
    timestamp = Column(Integer)  # field type: int
    x = Column(Float)  # type: float
    y = Column(Float)  # type: float
    z = Column(Float)  # type: float
    qw: float = Column(Float)
    qx: float = Column(Float)
    qy: float = Column(Float)
    qz: float = Column(Float)
    vx = Column(Float)  # type: float
    vy = Column(Float)  # type: float
    vz = Column(Float)  # type: float
    acceleration_x = Column(Float)  # type: float
    acceleration_y = Column(Float)  # type: float
    acceleration_z = Column(Float)  # type: float
    angular_rate_x = Column(Float)  # type: float
    angular_rate_y = Column(Float)  # type: float
    angular_rate_z = Column(Float)  # type: float
    epsg = Column(Integer)  # type: int
    log_token = Column(sql_types.HexLen8, ForeignKey("log.token"), nullable=False)  # type: str

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
    def quaternion(self) -> Quaternion:
        """
        Get the orientation of ego vehicle as quaternion respect to global coordinate system.
        :return: The orientation in quaternion.
        """
        return Quaternion(self.qw, self.qx, self.qy, self.qz)

    @property
    def translation_np(self) -> npt.NDArray[np.float64]:
        """
        Position of ego vehicle respect to global coordinate system.
        :return: <np.float: 3> Translation.
        """
        return np.array([self.x, self.y, self.z])

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

    def rotate_2d_points2d_to_ego_vehicle_frame(self, points2d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Rotate 2D points from global frame to ego-vehicle frame.
        :param points2d: <np.float: num_points, 2>. 2D points in global frame.
        :return: <np.float: num_points, 2>. 2D points rotated to ego-vehicle frame.
        """
        # Add zeros to the z dimension to make them 3D points.
        points3d: npt.NDArray[np.float32] = np.concatenate((points2d, np.zeros_like(points2d[:, 0:1])), axis=-1)
        # We need to extract the rotation around the z-axis only. since we are cropping a 2D map.
        # Construct scipy rotation instance using the rotation matrix from quaternion.
        rotation = R.from_matrix(self.quaternion.rotation_matrix.T)
        # Extract the angle of rotation around z-axis from the rotation.
        ego_rotation_angle = rotation.as_euler('zxy', degrees=True)[0]
        # Construct scipy rotation instance using ego_rotation_angle.
        xy_rotation = R.from_euler('z', ego_rotation_angle, degrees=True)
        # Rotate the corner points of the desired map crop to align with ego pose.
        rotated_points3d = xy_rotation.apply(points3d)
        # Remove the z dimension.
        rotated_points2d: npt.NDArray[np.float64] = rotated_points3d[:, :2]
        return rotated_points2d

    def get_map_crop(
        self,
        maps_db: Optional[GPKGMapsDB],
        xrange: Tuple[float, float],
        yrange: Tuple[float, float],
        map_layer_name: str,
        rotate_face_up: bool,
        target_imsize_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[npt.NDArray[np.float64]], npt.NDArray[np.float64], Tuple[float, ...]]:
        """
        This function returns the crop of the map centered at the current ego-pose with the given xrange and yrange.
        :param maps_db: Map database associated with this database.
        :param xrange: The range in x direction in meters relative to the current ego-pose. Eg: (-60, 60]).
        :param yrange: The range in y direction in meters relative to the current ego-pose Eg: (-60, 60).
        :param map_layer_name: A relevant map layer. Eg: 'drivable_area' or 'intensity'.
        :param rotate_face_up: Boolean indicating whether to rotate the image face up with respect to ego-pose.
        :param target_imsize_xy: The target grid xy dimensions for the output array. The xy resolution in meters / grid
            may be scaled by zooming to the desired dimensions.
        :return: (map_crop, map_translation, map_scale). Where:
            map_crop: The desired crop of the map.
            map_translation: The translation in map coordinates from the origin to the ego-pose.
            map_scale: Map scale (inverse of the map precision). This will be a tuple specifying the zoom in both the x
                and y direction if the target_imsize_xy parameter was set, which causes the resolution to change.

            map_scale and map_translation are useful for transforming objects like pointcloud/boxes to the map_crop.
            Refer to render_on_map().
        """
        if maps_db is None:
            precision: float = 1

            def to_pixel_coords(x: float, y: float) -> Tuple[float, float]:
                """
                Get the image coordinates given the x-y coordinates of point. This implementation simply returns the
                same coordinates.
                :param x: Global x coordinate.
                :param y: Global y coordinate.
                :return: Pixel coordinates in map.
                """
                return x, y

        else:
            map_layer = maps_db.load_layer(self.log.map_version, map_layer_name)
            precision = map_layer.precision
            to_pixel_coords = map_layer.to_pixel_coords

        map_scale: Tuple[float, ...] = (1.0 / precision, 1.0 / precision, 1.0)

        ego_translation = self.translation_np
        center_x, center_y = to_pixel_coords(ego_translation[0], ego_translation[1])
        center_x, center_y = int(center_x), int(center_y)

        top_left = int(xrange[0] * map_scale[0]), int(yrange[0] * map_scale[1])
        bottom_right = int(xrange[1] * map_scale[0]), int(yrange[1] * map_scale[1])

        # We need to extract the rotation around the z-axis only. since we are cropping a 2D map.
        # Construct scipy rotation instance using the rotation matrix from quaternion.
        rotation = R.from_matrix(self.quaternion.rotation_matrix.T)
        # Extract the angle of rotation around z-axis from the rotation.
        ego_rotation_angle = rotation.as_euler('zxy', degrees=True)[0]
        # Construct scipy rotation instance using ego_rotation_angle.

        xy_rotation = R.from_euler('z', ego_rotation_angle, degrees=True)

        map_rotate = 0

        # Rotate the corner points of the desired map crop to align with ego pose.
        rotated = xy_rotation.apply(
            [
                [top_left[0], top_left[1], 0],
                [top_left[0], bottom_right[1], 0],
                [bottom_right[0], top_left[1], 0],
                [bottom_right[0], bottom_right[1], 0],
            ]
        )[:, :2]

        # Construct minAreaRect using 4 corner points
        rect = cv2.minAreaRect(np.hstack([rotated[:, :1] + center_x, rotated[:, 1:] + center_y]).astype(int))
        rect_angle = rect[2]

        # Due to rounding error, the dimensions returned by cv2 may be off by 1, therefore it's better to manually
        # calculate the cropped dimensions instead of relying on the values returned by cv2 in rect[1]
        cropped_dimensions: npt.NDArray[np.float32] = np.array(
            [map_scale[0] * (xrange[1] - xrange[0]), map_scale[1] * (yrange[1] - yrange[0])]
        )
        rect = (rect[0], cropped_dimensions, rect_angle)

        # In OpenCV 4.4, the angle returned by cv2.minAreaRect is [-90,0). In OpenCV 4.5, the angle returned
        # appears to be [0, 90), though this isn't documented anywhere. To be compatible with both versions,
        # we adjust the angle to be [-90,0) if it isn't already.
        rect_angle = rect[2]
        cropped_dimensions = np.array([map_scale[0] * (xrange[1] - xrange[0]), map_scale[1] * (yrange[1] - yrange[0])])
        if rect_angle >= 0:
            rect = (rect[0], cropped_dimensions, rect_angle - 90)
        else:
            rect = (rect[0], cropped_dimensions, rect_angle)

        # We construct rect using cv2.minAreaRect, which takes only 4 unordered corner points, and can not consider
        # the angle of the required rect. The range of of 'angle' in cv2.minAreaRect is [-90,0).
        # A good explanation for the angle can be found at :
        # https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
        # Hence, we have to manually rotate the map after cropping based on the initial rotation angle.
        if ego_rotation_angle < -90:
            map_rotate = -90
        if -90 < ego_rotation_angle < 0:
            map_rotate = 0
        if 0 < ego_rotation_angle < 90:
            map_rotate = 90
        if 90 < ego_rotation_angle < 180:
            map_rotate = 180

        if map_layer is None:
            map_crop = None
        else:
            # Crop the rect using minAreaRect.
            map_crop = crop_rect(map_layer.data, rect)
            # Rotate the cropped map using adjusted angles,
            # since the angle is reset in cv2.minAreaRect every 90 degrees.
            map_crop = ndimage.rotate(map_crop, map_rotate, reshape=False)

            if rotate_face_up:
                # The map_crop is aligned with the ego_pose, but ego_pose is facing towards the right of the canvas,
                # but we need ego_pose to be facing up, hence rotating an extra 90 degrees.
                map_crop = np.rot90(map_crop)

        # These are in units of pixels, where x points to the right and y points *down*.
        if map_layer is None:
            map_upper_left_offset_from_global_coordinate_origin = np.zeros((2,))
        else:
            map_upper_left_offset_from_global_coordinate_origin = np.array(
                [-map_layer.transform_matrix[0, -1], map_layer.transform_matrix[1, -1]]
            )

        ego_offset_from_map_upper_left: npt.NDArray[np.float32] = np.array([center_x, -center_y])
        crop_upper_left_offset_from_ego: npt.NDArray[np.float32] = np.array(
            [xrange[0] * map_scale[0], yrange[0] * map_scale[1]]
        )
        map_translation: npt.NDArray[np.float64] = (
            -map_upper_left_offset_from_global_coordinate_origin
            - ego_offset_from_map_upper_left
            - crop_upper_left_offset_from_ego
        )
        map_translation_with_z: npt.NDArray[np.float64] = np.array(
            [map_translation[0], map_translation[1], 0]
        )  # add z-coordinate

        if target_imsize_xy is not None:
            zoom_size_x = target_imsize_xy[0] / cropped_dimensions[0]
            zoom_size_y = target_imsize_xy[1] / cropped_dimensions[1]
            map_crop = ndimage.zoom(map_crop, [zoom_size_x, zoom_size_y])
            map_scale = (zoom_size_x, zoom_size_y)

        return map_crop, map_translation_with_z, map_scale

    def get_vector_map(
        self,
        maps_db: Optional[GPKGMapsDB],
        xrange: Tuple[float, float],
        yrange: Tuple[float, float],
        connection_scales: Optional[List[int]] = None,
    ) -> VectorMapNp:
        """
        This function returns the crop of baseline paths (blps) map centered at the current ego-pose with
        the given xrange and yrange.
        :param maps_db: Map database associated with this database.
        :param xrange: The range in x direction in meters relative to the current ego-pose. Eg: [-60, 60].
        :param yrange: The range in y direction in meters relative to the current ego-pose Eg: [-60, 60].
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        # load geopandas data
        map_version = self.lidar_pc.log.map_version.replace('.gpkg', '')
        blps_gdf = maps_db.load_vector_layer(map_version, 'baseline_paths')  # type: ignore
        lane_poly_gdf = maps_db.load_vector_layer(map_version, 'lanes_polygons')  # type: ignore
        intersections_gdf = maps_db.load_vector_layer(map_version, 'intersections')  # type: ignore
        lane_connectors_gdf = maps_db.load_vector_layer(map_version, 'lane_connectors')  # type: ignore
        lane_groups_gdf = maps_db.load_vector_layer(map_version, 'lane_groups_polygons')  # type: ignore

        if (
            (blps_gdf is None)
            or (lane_poly_gdf is None)
            or (intersections_gdf is None)
            or (lane_connectors_gdf is None)
            or (lane_groups_gdf is None)
        ):
            # This sample has no vector map.
            coords: npt.NDArray[np.float32] = np.empty([0, 2, 2], dtype=np.float32)
            if not connection_scales:
                # Use the 1-hop connections if connection_scales is not specified.
                connection_scales = [1]
            multi_scale_connections: Dict[int, Any] = {
                scale: np.empty([0, 2], dtype=np.int64) for scale in connection_scales
            }
            return VectorMapNp(
                coords=coords,
                multi_scale_connections=multi_scale_connections,
            )

        # data enhancement
        blps_in_lanes = blps_gdf[blps_gdf['lane_fid'].notna()]
        blps_in_intersections = blps_gdf[blps_gdf['lane_connector_fid'].notna()]

        # enhance blps_in_lanes
        lane_group_info = lane_poly_gdf[['lane_fid', 'lane_group_fid']]
        blps_in_lanes = blps_in_lanes.merge(lane_group_info, on='lane_fid', how='outer')

        # enhance blps_in_intersections
        lane_connectors_gdf['lane_connector_fid'] = lane_connectors_gdf['fid']
        lane_conns_info = lane_connectors_gdf[
            ['lane_connector_fid', 'intersection_fid', 'exit_lane_fid', 'entry_lane_fid']
        ]
        # Convert the exit_fid field of both data frames to the same dtype for merging.
        lane_conns_info = lane_conns_info.astype({'lane_connector_fid': int})
        blps_in_intersections = blps_in_intersections.astype({'lane_connector_fid': int})
        blps_in_intersections = blps_in_intersections.merge(lane_conns_info, on='lane_connector_fid', how='outer')

        # enhance blps_connection info
        lane_blps_info = blps_in_lanes[['fid', 'lane_fid']]
        from_blps_info = lane_blps_info.rename(columns={'fid': 'from_blp', 'lane_fid': 'exit_lane_fid'})
        to_blps_info = lane_blps_info.rename(columns={'fid': 'to_blp', 'lane_fid': 'entry_lane_fid'})
        blps_in_intersections = blps_in_intersections.merge(from_blps_info, on='exit_lane_fid', how='inner')
        blps_in_intersections = blps_in_intersections.merge(to_blps_info, on='entry_lane_fid', how='inner')

        # Select in-range blps
        candidate_lane_groups, candidate_intersections = get_candidates(
            self.translation_np, xrange, yrange, lane_groups_gdf, intersections_gdf
        )
        candidate_blps_in_lanes = blps_in_lanes[
            blps_in_lanes['lane_group_fid'].isin(candidate_lane_groups['fid'].astype(int))
        ]
        candidate_blps_in_intersections = blps_in_intersections[
            blps_in_intersections['intersection_fid'].isin(candidate_intersections['fid'].astype(int))
        ]

        ls_coordinates_list: List[List[List[float]]] = []
        ls_connections_list: List[List[int]] = []
        ls_groupings_list: List[List[int]] = []
        cross_blp_connection: Dict[str, List[int]] = dict()

        # generate lane_segments from blps in lanes
        build_lane_segments_from_blps(
            candidate_blps_in_lanes, ls_coordinates_list, ls_connections_list, ls_groupings_list, cross_blp_connection
        )
        # generate lane_segments from blps in intersections
        build_lane_segments_from_blps(
            candidate_blps_in_intersections,
            ls_coordinates_list,
            ls_connections_list,
            ls_groupings_list,
            cross_blp_connection,
        )

        # generate connections between blps
        for blp_id, blp_info in cross_blp_connection.items():
            # Add predecessors
            connect_blp_predecessor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_connections_list)
            # Add successors
            connect_blp_successor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_connections_list)

        ls_coordinates: npt.NDArray[np.float64] = np.asarray(ls_coordinates_list, self.translation_np.dtype)
        ls_connections: npt.NDArray[np.int64] = np.asarray(ls_connections_list, np.int64)
        # Transform the lane coordinates from global frame to ego vehicle frame.
        # Flatten ls_coordinates from (num_ls, 2, 2) to (num_ls * 2, 2) for easier processing.
        ls_coordinates = ls_coordinates.reshape(-1, 2)
        ls_coordinates = ls_coordinates - self.translation_np[:2]
        ls_coordinates = self.rotate_2d_points2d_to_ego_vehicle_frame(ls_coordinates)
        ls_coordinates = ls_coordinates.reshape(-1, 2, 2).astype(np.float32)

        if connection_scales:
            # Generate multi-scale connections.
            multi_scale_connections = generate_multi_scale_connections(ls_connections, connection_scales)
        else:
            # Use the 1-hop connections if connection_scales is not specified.
            multi_scale_connections = {1: ls_connections}

        return VectorMapNp(
            coords=ls_coordinates,
            multi_scale_connections=multi_scale_connections,
        )
