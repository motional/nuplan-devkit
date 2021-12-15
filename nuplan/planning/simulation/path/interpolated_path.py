from typing import Callable, List

import numpy as np
import scipy.interpolate as sp_interp
from nuplan.common.actor_state.state_representation import ProgressStateSE2
from nuplan.planning.simulation.path.path import AbstractPath


class InterpolatedPath(AbstractPath):
    def __init__(self, path: List[ProgressStateSE2]):
        """
        A path that is interpolated from a list of points.

        :param path: List of states creating a path.
            The path has to have at least 2 elements, otherwise it is considered invalid and the class will raise.
        """
        assert len(path) > 1, "Path has to has more than 1 element!"
        self._path = path

        # Re-arrange to arrays for interpolation
        progress = [point.progress for point in self._path]
        array_states = np.array([list(point) for point in self._path], dtype='float64')
        self.function_interp: Callable[[float], ProgressStateSE2] = sp_interp.interp1d(progress, array_states, axis=0)

    def get_start_progress(self) -> float:
        """ Inherited, see superclass. """
        return self._path[0].progress  # type: ignore

    def get_end_progress(self) -> float:
        """ Inherited, see superclass. """
        return self._path[-1].progress  # type: ignore

    def get_state_at_progress(self, progress: float) -> ProgressStateSE2:
        """ Inherited, see superclass. """
        self._assert_progress(progress)

        interpolated_array = self.function_interp(progress)
        return ProgressStateSE2.deserialize(list(interpolated_array))

    def get_sampled_path(self) -> List[ProgressStateSE2]:
        """ Inherited, see superclass. """
        return self._path

    def _assert_progress(self, progress: float) -> None:
        """ Check if queried progress is within bounds """
        start_progress = self.get_start_progress()
        end_progress = self.get_end_progress()
        assert start_progress <= progress <= end_progress, f"Progress exceeds path! " \
            f"{start_progress} <= {progress} <= {end_progress}"
