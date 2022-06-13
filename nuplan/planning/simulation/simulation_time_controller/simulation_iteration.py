from dataclasses import dataclass

from nuplan.common.actor_state.state_representation import TimePoint


@dataclass
class SimulationIteration:
    """
    Simulation step time and index.
    """

    time_point: TimePoint  # A time point along simulation

    # Iteration in the simulation, starting from 0.
    # In open loop this represents the n-th sample in a scenario from the log.
    # In closed loop this represents the n-th sample of the simulation.
    index: int

    def __post_init__(self) -> None:
        """Post-init index sanity check."""
        assert self.index >= 0, f"Iteration must be >= 0, but it is {self.index}!"

    @property
    def time_us(self) -> int:
        """
        :return: time in micro seconds.
        """
        return int(self.time_point.time_us)

    @property
    def time_s(self) -> float:
        """
        :return: Time in seconds.
        """
        return float(self.time_point.time_s)
