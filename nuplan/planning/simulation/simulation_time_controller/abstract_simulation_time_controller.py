from __future__ import annotations  # Used to avoid circular dependency between A.Scenario and A.SimulationManager

import abc
from typing import Optional

from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class AbstractSimulationTimeController(abc.ABC):
    """
    Generic simulation time manager.
    """

    @abc.abstractmethod
    def get_iteration(self) -> SimulationIteration:
        """
        Get the current simulation iteration.
        :return: Get the current simulation current_simulation_state and time point
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the observation (all internal states should be reseted, if any).
        """
        pass

    @abc.abstractmethod
    def next_iteration(self) -> Optional[SimulationIteration]:
        """
        Move to the next iteration and return its simulation iteration.
        Returns None if we have reached the end of the simulation.
        """
        pass

    @abc.abstractmethod
    def reached_end(self) -> bool:
        """
        Check if we have reached the end of the simulation.
        :return: Check whether simulation reached the end state.
        """
        pass

    @abc.abstractmethod
    def number_of_iterations(self) -> int:
        """
        The number of iterations the simulation should be running for
        :return: Number of iterations of simulation.
        """
        pass
