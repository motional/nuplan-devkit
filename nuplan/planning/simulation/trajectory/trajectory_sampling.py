from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, cast

PROXIMITY_ABS_TOL = 1e-10


@dataclass
class TrajectorySampling:
    """
    Trajectory sampling config. The variables are set as optional, to make sure we can deduce last variable if only
        two are set.
    """

    # Number of poses in trajectory in addition to initial state
    num_poses: Optional[int] = None
    # [s] the time horizon of a trajectory
    time_horizon: Optional[float] = None
    # [s] length of an interval between two states
    interval_length: Optional[float] = None

    def __post_init__(self) -> None:
        """
        Make sure all entries are correctly initialized.
        """
        if self.num_poses and not isinstance(self.num_poses, int):
            raise ValueError(f"num_poses was defined but it is not int. Instead {type(self.num_poses)}!")
        if self.time_horizon:
            self.time_horizon = float(self.time_horizon)
        if self.interval_length:
            self.interval_length = float(self.interval_length)
        if self.num_poses and self.time_horizon and not self.interval_length:
            self.interval_length = self.time_horizon / self.num_poses
        elif self.num_poses and self.interval_length and not self.time_horizon:
            self.time_horizon = self.num_poses * self.interval_length
        elif self.time_horizon and self.interval_length and not self.num_poses:
            remainder = math.fmod(self.time_horizon, self.interval_length)
            is_close_to_zero = math.isclose(remainder, 0, abs_tol=PROXIMITY_ABS_TOL)
            is_close_to_interval_length = math.isclose(remainder, self.interval_length, abs_tol=PROXIMITY_ABS_TOL)
            if not is_close_to_zero and not is_close_to_interval_length:
                raise ValueError(
                    "The time horizon must be a multiple of interval length! "
                    f"time_horizon = {self.time_horizon}, interval = {self.interval_length} and is {remainder}"
                )
            self.num_poses = int(self.time_horizon / self.interval_length)
        elif self.num_poses and self.time_horizon and self.interval_length:
            if not math.isclose(self.num_poses, self.time_horizon / self.interval_length, abs_tol=PROXIMITY_ABS_TOL):
                raise ValueError(
                    "Not valid initialization of sampling class!"
                    f"time_horizon = {self.time_horizon}, "
                    f"interval = {self.interval_length}, num_poses = {self.num_poses}"
                )

        else:
            raise ValueError(
                f"Cant initialize class! num_poses = {self.num_poses}, "
                f"interval = {self.interval_length}, time_horizon = {self.time_horizon}"
            )

    @property
    def step_time(self) -> float:
        """
        :return: [s] The time difference between two poses.
        """
        if not self.interval_length:
            raise RuntimeError("Invalid interval length!")
        return self.interval_length

    def __hash__(self) -> int:
        """
        :return: hash for the dataclass. It has to be custom because the dataclass is not frozen.
            It is not frozen because we deduce the missing parameters.
        """
        return hash((self.num_poses, self.time_horizon, self.interval_length))

    def __eq__(self, other: object) -> bool:
        """
        Compare two instances of trajectory sampling
        :param other: object, needs to be TrajectorySampling class
        :return: true, if they are equal, false otherwise
        """
        if not isinstance(other, TrajectorySampling):
            return NotImplemented
        return (
            math.isclose(cast(float, other.time_horizon), cast(float, self.time_horizon))
            and math.isclose(cast(float, other.interval_length), cast(float, self.interval_length))
            and other.num_poses == self.num_poses
        )
