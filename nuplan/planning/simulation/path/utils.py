from typing import List

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import ProgressStateSE2, StateSE2
from nuplan.planning.simulation.path.path import AbstractPath


def calculate_progress(path: List[StateSE2]) -> List[float]:
    """
    Calculate the cumulative progress of a given path

    :param path: a path consisting of StateSE2 as waypoints
    :return: a cumulative list of progress
    """
    x_position = [point.x for point in path]
    y_position = [point.y for point in path]
    x_diff = np.diff(x_position)
    y_diff = np.diff(y_position)
    points_diff: npt.NDArray[np.float_] = np.concatenate(([x_diff], [y_diff]), axis=0)
    progress_diff = np.append(0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff).tolist()  # type: ignore


def convert_se2_path_to_progress_path(path: List[StateSE2]) -> List[ProgressStateSE2]:
    """
    Converts a list of StateSE2 to a list of ProgressStateSE2

    :return: a list of ProgressStateSE2
    """
    progress_list = calculate_progress(path)
    return [
        ProgressStateSE2(progress=progress, x=point.x, y=point.y, heading=point.heading)
        for point, progress in zip(path, progress_list)
    ]


def trim_path_up_to_progress(path: AbstractPath, progress: float) -> List[ProgressStateSE2]:
    """
    Returns a trimmed path where the starting pose is starts at the given progress. Everything before is discarded
    :param path: the path to be trimmed
    :param progress: the progress where the path should start.
    :return: the trimmed discrete sampled path starting from the given progress
    """
    start_progress = path.get_start_progress()
    end_progress = path.get_end_progress()
    assert start_progress <= progress <= end_progress, (
        f"Progress exceeds path! " f"{start_progress} <= {progress} <= {end_progress}"
    )

    cut_path = [path.get_state_at_progress(progress)]  # add state at cut_off point
    progress_list: npt.NDArray[np.float_] = np.array([point.progress for point in path.get_sampled_path()])
    trim_indices = np.argwhere(progress_list > progress)

    if trim_indices.size > 0:
        trim_index = trim_indices.flatten()[0]
        cut_path += path.get_sampled_path()[trim_index:]
        return cut_path

    return path.get_sampled_path()[-2:]  # type: ignore


def trim_path(path: AbstractPath, start: float, end: float) -> List[ProgressStateSE2]:
    """
    Returns a trimmed path to be between given start and end progress. Everything else is discarded.
    :param path: the path to be trimmed
    :param start: the progress where the path should start.
    :param end: the progress where the path should end.
    :return: the trimmed discrete sampled path starting and ending from the given progress
    """
    start_progress = path.get_start_progress()
    end_progress = path.get_end_progress()
    assert start <= end, f"Start progress has to be less than the end progress {start} <= {end}"
    assert start_progress <= start, f"Start progress exceeds path! {start_progress} <= {start}"
    assert end <= end_progress, f"End progress exceeds path! {end} <= {end_progress}"

    start_state, end_state = path.get_state_at_progresses([start, end])  # interpolated state at start point

    progress_list: npt.NDArray[np.float_] = np.array([point.progress for point in path.get_sampled_path()])
    trim_front_indices = np.argwhere(progress_list > start)
    trim_tail_indices = np.argwhere(progress_list < end)
    if trim_front_indices.size > 0:
        trim_front_index = trim_front_indices.flatten()[0]
    else:
        # Account for the case that the queried start progress == end_progress
        return path.get_sampled_path()[-2:]  # type: ignore

    if trim_tail_indices.size > 0:
        trim_end_index = trim_tail_indices.flatten()[-1]
    else:
        # Account for the case that the queried end progress == start_progress
        return path.get_sampled_path()[:2]  # type: ignore

    return [start_state] + path.get_sampled_path()[trim_front_index : trim_end_index + 1] + [end_state]  # type: ignore
