from typing import List

import numpy as np
import scipy.interpolate as sp_interp

from nuplan.common.actor_state.state_representation import ProgressStateSE2
from nuplan.common.geometry.compute import AngularInterpolator
from nuplan.planning.simulation.path.path import AbstractPath


class InterpolatedPath(AbstractPath):
    """A path that is interpolated from a list of points."""

    def __init__(self, path: List[ProgressStateSE2]):
        """
        Constructor of InterpolatedPath.

        :param path: List of states creating a path.
            The path has to have at least 2 elements, otherwise it is considered invalid and the class will raise.
        """
        assert len(path) > 1, "Path has to has more than 1 element!"
        self._path = path

        # Re-arrange to arrays for interpolation
        progress = [point.progress for point in self._path]

        linear_states = []
        angular_states = []
        for point in path:
            linear_states.append([point.progress, point.x, point.y])
            angular_states.append([point.heading])

        linear_states = np.array(linear_states, dtype='float64')  # type: ignore
        angular_states = np.array(angular_states, dtype='float64')  # type: ignore
        self._function_interp_linear = sp_interp.interp1d(progress, linear_states, axis=0)
        self._angular_interpolator = AngularInterpolator(progress, angular_states)

    def get_start_progress(self) -> float:
        """Inherited, see superclass."""
        return self._path[0].progress  # type: ignore

    def get_end_progress(self) -> float:
        """Inherited, see superclass."""
        return self._path[-1].progress  # type: ignore

    def get_state_at_progress(self, progress: float) -> ProgressStateSE2:
        """Inherited, see superclass."""
        self._assert_progress(progress)

        linear_states = list(self._function_interp_linear(progress))
        angular_states = list(self._angular_interpolator.interpolate(progress))

        return ProgressStateSE2.deserialize(linear_states + angular_states)

    def get_state_at_progresses(self, progresses: List[float]) -> List[ProgressStateSE2]:
        """Inherited, see superclass."""
        self._assert_progress(min(progresses))
        self._assert_progress(max(progresses))

        linear_states_batch = self._function_interp_linear(progresses)
        angular_states_batch = self._angular_interpolator.interpolate(progresses)

        return [
            ProgressStateSE2.deserialize(list(linear_states) + list(angular_states))
            for linear_states, angular_states in zip(linear_states_batch, angular_states_batch)
        ]

    def get_sampled_path(self) -> List[ProgressStateSE2]:
        """Inherited, see superclass."""
        return self._path

    def _assert_progress(self, progress: float) -> None:
        """Check if queried progress is within bounds"""
        start_progress = self.get_start_progress()
        end_progress = self.get_end_progress()
        assert start_progress <= progress <= end_progress, (
            f"Progress exceeds path! " f"{start_progress} <= {progress} <= {end_progress}"
        )
