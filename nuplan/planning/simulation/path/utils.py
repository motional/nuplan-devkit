from typing import List

import numpy as np
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
    points_diff = np.concatenate(([x_diff], [y_diff]), axis=0)
    progress_diff = np.append(0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff).tolist()  # type: ignore


def convert_se2_path_to_progress_path(path: List[StateSE2]) -> List[ProgressStateSE2]:
    """
    Converts a list of StateSE2 to a list of ProgressStateSE2

    :return: a list of ProgressStateSE2
    """
    progress_list = calculate_progress(path)
    return [ProgressStateSE2(progress=progress, x=point.x, y=point.y, heading=point.heading)
            for point, progress in zip(path, progress_list)]


def get_trimmed_path_up_to_progress(path: AbstractPath, progress: float) -> List[ProgressStateSE2]:
    """
    Returns a trimmed path where the starting pose is starts at the given progress. Everything before is discarded
    :param path: the path to be trimmed
    :param progress: the progress where the path should start.
    :return: the trimmed discrete sampled path starting from the given progress
    """
    start_progress = path.get_start_progress()
    end_progress = path.get_end_progress()
    assert start_progress <= progress <= end_progress, f"Progress exceeds path! " \
                                                       f"{start_progress} <= {progress} <= {end_progress}"

    trimmed_path = [path.get_state_at_progress(progress)]  # add state at cut_off point
    progress_list = np.array([point.progress for point in path.get_sampled_path()])
    trim_indices = np.argwhere(progress_list > progress)

    if trim_indices.size > 0:
        trim_index = trim_indices.flatten()[0]
        trimmed_path += path.get_sampled_path()[trim_index:]
        return trimmed_path

    return path.get_sampled_path()[-2:]  # type: ignore
