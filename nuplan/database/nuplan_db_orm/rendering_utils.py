from __future__ import annotations  # postpone evaluation of annotations

import logging
from functools import reduce
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import PIL.Image
from matplotlib.axes import Axes
from sqlalchemy import func

from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.database.nuplan_db_orm.image import Image
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.utils.boxes.box3d import BoxVisibility, box_in_image
from nuplan.database.utils.geometry import view_points

logger = logging.getLogger()


def lidar_pc_closest_image(lidar_pc: LidarPc, camera_channels: Optional[List[str]] = None) -> List[Image]:
    """
    Find the closest images to LidarPc.
    :param camera_channels: List of image channels to find closest image of.
    :return: List of Images from the provided channels closest to LidarPc.
    """
    if camera_channels is None:
        camera_channels = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_R0", "CAM_R1"]

    imgs = []
    for channel in camera_channels:
        img = (
            lidar_pc._session.query(Image)
            .join(Camera)
            .filter(Image.camera_token == Camera.token)
            .filter(Camera.channel == channel)
            .filter(Camera.log_token == lidar_pc.lidar.log_token)
            .order_by(func.abs(Image.timestamp - lidar_pc.timestamp))
            .first()
        )
        imgs.append(img)

    return imgs


def render_pointcloud_in_image(
    db: NuPlanDB,
    lidar_pc: LidarPc,
    dot_size: int = 5,
    color_channel: int = 2,
    max_radius: float = np.inf,
    image_channel: str = "CAM_F0",
) -> None:
    """
    Scatter-plots pointcloud on top of image.
    :param db: Log Database.
    :param sample: LidarPc Sample.
    :param dot_size: Scatter plot dot size.
    :param color_channel: Set to 2 for coloring dots by height, 3 for intensity.
    :param max_radius: Max xy radius of lidar points to include in visualization.
        Set to np.inf to include all points.
    :param image_channel: Which image to render.
    """
    image = lidar_pc_closest_image(lidar_pc, [image_channel])[0]

    points, coloring, im = map_pointcloud_to_image(
        db, lidar_pc, image, color_channel=color_channel, max_radius=max_radius
    )
    plt.figure(figsize=(9, 16))
    plt.imshow(im)
    plt.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    plt.axis("off")


def map_pointcloud_to_image(
    db: NuPlanDB, lidar_pc: LidarPc, img: Image, color_channel: int = 2, max_radius: float = np.inf
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], PIL.Image.Image]:
    """
    Given a lidar and camera sample_data, load point-cloud and map it to the image plane.
    :param db: Log Database.
    :param lidar_pc: Lidar sample_data record.
    :param img: Camera sample_data record.
    :param color_channel: Set to 2 for coloring dots by depth, 3 for intensity.
    :param max_radius: Max xy radius of lidar points to include in visualization.
        Set to np.inf to include all points.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    assert isinstance(lidar_pc, LidarPc), "first input must be a lidar_pc modality"
    assert isinstance(img, Image), "second input must be a camera modality"

    # Load files.
    pc = lidar_pc.load()
    im = img.load_as(db, img_type="pil")

    # Filter lidar points to be inside desired range.
    radius = np.sqrt(pc.points[0] ** 2 + pc.points[1] ** 2)
    keep = radius <= max_radius
    pc.points = pc.points[:, keep]

    # Transform pc to img.
    transform = reduce(
        np.dot,
        [
            img.camera.trans_matrix_inv,
            img.ego_pose.trans_matrix_inv,
            lidar_pc.ego_pose.trans_matrix,
            lidar_pc.lidar.trans_matrix,
        ],
    )
    pc.transform(transform)

    # Grab the coloring (depth or intensity).
    coloring = pc.points[color_channel, :]
    depths = pc.points[2, :]

    # Take the actual picture (matrix multiplication with camera - matrix + renormalization).
    points = view_points(pc.points[:3, :], img.camera.intrinsic_np, normalize=True)

    # Finally filter away points outside the image.
    mask: npt.NDArray[np.bool8] = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, im


def render_lidar_box(lidar_box: LidarBox, db: NuPlanDB, ax: Optional[List[Axes]] = None) -> None:
    """
    Render LidarBox on an image and a lidar.
    :param lidar_box: A LidarBox object
    :param db: Log Database.
    :param ax: Array of Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))

    pc = lidar_box.lidar_pc
    imgs = lidar_pc_closest_image(lidar_box.lidar_pc)

    # Figure out which camera the object is visible in (may return nothing)
    found = False
    for img in imgs:
        cam = img.camera
        box = lidar_box.box()
        box.transform(img.ego_pose.trans_matrix_inv)  # Move box to ego vehicle coord system
        box.transform(cam.trans_matrix_inv)  # Move box to sensor coord system
        if box_in_image(box, cam.intrinsic_np, (cam.width, cam.height), vis_level=BoxVisibility.ANY):
            found = True
            break  # Found an image that matches.

    assert found, "Could not find image where annotation is visible"

    # Get the color
    if not lidar_box.category:
        logger.error("Wrong 3d instance mapping", lidar_box)
        c: npt.NDArray[np.float64] = np.array([128, 0, 128]) / 255.0
    else:
        c = lidar_box.category.color_np

    color = c, c, np.array([0, 0, 0])  # type: ignore

    # === CAMERA view ===
    ax[0].imshow(img.load_as(db, img_type="pil"))
    box.render(ax[0], view=img.camera.intrinsic_np, normalize=True, colors=color)
    ax[0].set_title(img.camera.channel)
    ax[0].axis("off")
    ax[0].set_aspect("equal")

    # === LIDAR view ===
    box = lidar_box.box()  # Need to re-load box from the global coord-system.
    box.transform(pc.ego_pose.trans_matrix_inv)  # Move box to ego vehicle coord system
    box.transform(pc.lidar.trans_matrix_inv)  # Move box to sensor coord system

    view = np.eye(4)
    pc.load(db).render_height(ax[1], view=view)
    box.render(ax[1], view=view, colors=color)

    corners = view_points(box.corners(), view, False)[:2, :]
    ax[1].set_xlim([np.amin(corners[0, :]) - 10, np.amax(corners[0, :]) + 10])
    ax[1].set_ylim([np.amin(corners[1, :]) - 10, np.amax(corners[1, :]) + 10])
    ax[1].axis("off")
    ax[1].set_aspect("equal")
