from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectorySampling:
    num_poses: int  # Number of poses in trajectory in addition to initial state
    time_horizon: float  # [s] the time horizon of a trajectory

    @property
    def step_time(self) -> float:
        """
        :return: [s] the time difference between two poses
        """
        return self.time_horizon / self.num_poses
