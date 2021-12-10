from enum import Enum
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class Color(Enum):
    BACKGROUND: Tuple[float, float, float] = (0.31, 0.31, 0.31)
    ROADMAP: Tuple[float, float, float] = (0.82, 0.82, 0.82)
    AGENTS: Tuple[float, float, float] = (0.0, 0.55, 0.55)
    EGO: Tuple[float, float, float] = (0.72, 0.53, 0.04)
    PREDICTIONS: Tuple[float, float, float] = (0.55, 0.0, 0.0)
    BASELINE_PATHS: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def get_raster_with_trajectories_as_rgb(
        pixel_size: float,
        raster: Raster,
        target_trajectory: Optional[Trajectory] = None,
        predicted_trajectory: Optional[Trajectory] = None,
        background_color: Color = Color.BACKGROUND,
        roadmap_color: Color = Color.ROADMAP,
        agents_color: Color = Color.AGENTS,
        ego_color: Color = Color.EGO,
        target_trajectory_color: Color = Color.EGO,
        predicted_trajectory_color: Color = Color.PREDICTIONS,
        baseline_paths_color: Color = Color.BASELINE_PATHS,
) -> npt.NDArray[np.float32]:
    """
    Creates an RGB images of the raster layers overlayed with predicted / ground truth trajectories

    :param raster: input raster to visualize
    :param target_trajectory: target (ground truth) trajectory to visualize
    :param predicted_trajectory: predicted trajectory to visualize
    :param background_color: desired color of the image's background
    :param roadmap_color: desired color of the map raster layer
    :param agents_color: desired color of the agents raster layer
    :param ego_color: desired color of the ego raster layer
    :param target_trajectory_color: desired color of the target trajectory
    :param predicted_trajectory_color: desired color of the predicted trajectory
    :param pixel_size: [m] size of pixel in meters
    :return: constructed RGB image
    """
    grid_shape = (raster.height, raster.width)
    image = np.full((*grid_shape, 3), background_color.value, dtype=np.float32)

    image[raster.roadmap_layer.nonzero()] = roadmap_color.value
    image[raster.agents_layer.nonzero()] = agents_color.value
    image[raster.ego_layer.nonzero()] = ego_color.value
    image[raster.baseline_paths_layer.nonzero()] = baseline_paths_color.value

    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, target_trajectory_color, pixel_size)

    if predicted_trajectory is not None:
        _draw_trajectory(image, predicted_trajectory, predicted_trajectory_color, pixel_size)

    return image


def _draw_trajectory(image: npt.NDArray[np.float32], trajectory: Trajectory, color: Color, pixel_size: float) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2

    pixels_x = (center_x - (trajectory.numpy_position_x / pixel_size)).astype(np.int32)
    pixels_y = (center_y - (trajectory.numpy_position_y / pixel_size)).astype(np.int32)
    pixels_x = np.concatenate([pixels_x, pixels_x + 1, pixels_x - 1, pixels_x, pixels_x])  # type: ignore
    pixels_y = np.concatenate([pixels_y, pixels_y, pixels_y, pixels_y + 1, pixels_y - 1])  # type: ignore
    idxs = np.logical_and.reduce([0 <= pixels_x, pixels_x < grid_width, 0 <= pixels_y, pixels_y < grid_height])

    image[(pixels_x[idxs], pixels_y[idxs])] = color.value
