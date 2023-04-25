from __future__ import annotations

from typing import Any, List, Tuple, Type

import numpy as np
import scipy.interpolate as sp_interp

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.geometry.compute import AngularInterpolator
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.utils.split_state import SplitState
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class InterpolatedTrajectory(AbstractTrajectory):
    """Class representing a trajectory that can be interpolated from a list of points."""

    def __init__(self, trajectory: List[InterpolatableState]):
        """
        :param trajectory: List of states creating a trajectory.
            The trajectory has to have at least 2 elements, otherwise it is considered invalid and the class will raise.
        """
        assert trajectory, "Trajectory can't be empty!"
        assert isinstance(trajectory[0], InterpolatableState)
        self._trajectory_class = trajectory[0].__class__
        assert all(isinstance(point, self._trajectory_class) for point in trajectory)
        if len(trajectory) <= 1:
            raise ValueError(f"There is not enough states in trajectory: {len(trajectory)}!")

        self._trajectory = trajectory

        # Time points for interpolation axis
        time_series = [point.time_us for point in trajectory]

        # Split in linear states and angular states
        linear_states = []
        angular_states = []
        for point in trajectory:
            split_state = point.to_split_state()
            linear_states.append(split_state.linear_states)
            angular_states.append(split_state.angular_states)

        self._fixed_state = trajectory[0].to_split_state().fixed_states

        linear_states = np.array(linear_states, dtype='float64')  # type: ignore
        angular_states = np.array(angular_states, dtype='float64')  # type: ignore
        self._function_interp_linear = sp_interp.interp1d(time_series, linear_states, axis=0)
        self._angular_interpolator = AngularInterpolator(time_series, angular_states)

    def __reduce__(self) -> Tuple[Type[InterpolatedTrajectory], Tuple[Any, ...]]:
        """
        Helper for pickling.
        """
        return self.__class__, (self._trajectory,)

    @property
    def start_time(self) -> TimePoint:
        """Inherited, see superclass."""
        return self._trajectory[0].time_point

    @property
    def end_time(self) -> TimePoint:
        """Inherited, see superclass."""
        return self._trajectory[-1].time_point

    def get_state_at_time(self, time_point: TimePoint) -> InterpolatableState:
        """Inherited, see superclass."""
        start_time = self.start_time
        end_time = self.end_time

        assert start_time <= time_point <= end_time, (
            f"Interpolation time {time_point=} not in trajectory time window! \n"
            f"{start_time.time_us=} <= {time_point.time_us=} <= {end_time.time_us=}"
        )

        linear_states = list(self._function_interp_linear(time_point.time_us))
        angular_states = list(self._angular_interpolator.interpolate(time_point.time_us))

        return self._trajectory_class.from_split_state(SplitState(linear_states, angular_states, self._fixed_state))

    def get_state_at_times(self, time_points: List[TimePoint]) -> List[InterpolatableState]:
        """Inherited, see superclass."""
        start_time = self.start_time
        end_time = self.end_time

        assert start_time <= min(time_points), (
            f"Interpolation time not in trajectory time window! The following is not satisfied:"
            f"Trajectory start time: ({start_time.time_s}) <= Earliest interpolation time ({min(time_points).time_s}) "
            f"{max(time_points).time_s} <= {end_time.time_s} "
        )

        assert max(time_points) <= end_time, (
            f"Interpolation time not in trajectory time window! The following is not satisfied:"
            f"Trajectory end time: ({end_time.time_s}) >= Latest interpolation time ({max(time_points).time_s}) "
        )

        interpolation_times = [t.time_us for t in time_points]
        linear_states = list(self._function_interp_linear(interpolation_times))
        angular_states = list(self._angular_interpolator.interpolate(interpolation_times))

        return [
            self._trajectory_class.from_split_state(SplitState(lin_state, ang_state, self._fixed_state))
            for lin_state, ang_state in zip(linear_states, angular_states)
        ]

    def get_sampled_trajectory(self) -> List[InterpolatableState]:
        """Inherited, see superclass."""
        return self._trajectory
