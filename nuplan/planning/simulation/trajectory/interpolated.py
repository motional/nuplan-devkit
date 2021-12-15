from typing import List

import numpy as np
import scipy.interpolate as sp_interp
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory


class InterpolatedTrajectory(AbstractTrajectory):

    def __init__(self, trajectory: List[EgoState]):
        """
        A trajectory that is interpolated from a list of points.

        :param trajectory: List of states creating a trajectory.
            The trajectory has to have at least 2 elements, otherwise it is considered invalid and the class will raise.
        """
        assert len(trajectory) > 1, "Trajectory has to has more than 1 element!"
        self.trajectory = trajectory

        # Re-arrange to arrays for interpolation
        time_series = [point.time_us for point in self.trajectory]
        array_states = np.array([list(point) for point in self.trajectory], dtype='float64')
        self.function_interp = sp_interp.interp1d(time_series, array_states, axis=0)

    @property
    def start_time(self) -> TimePoint:
        """ Inherited, see superclass. """
        return self.trajectory[0].time_point

    @property
    def end_time(self) -> TimePoint:
        """ Inherited, see superclass. """
        return self.trajectory[-1].time_point

    def get_state_at_time(self, time_point: TimePoint) -> EgoState:
        """ Inherited, see superclass. """
        start_time = self.start_time
        end_time = self.end_time
        assert start_time <= time_point <= end_time, f"Timeout exceeds trajectory! " \
                                                     f"{start_time.time_s} <= {time_point.time_s} <= {end_time.time_s}"

        interpolated_array = self.function_interp(time_point.time_us)
        return EgoState.deserialize(interpolated_array)

    def get_sampled_trajectory(self) -> List[EgoState]:
        """ Inherited, see superclass. """
        return self.trajectory
