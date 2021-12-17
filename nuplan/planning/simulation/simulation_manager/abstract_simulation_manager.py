from __future__ import annotations  # Used to avoid circular dependency between A.Scenario and A.SimulationManager

from abc import ABCMeta, abstractmethod
from typing import Optional

from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration


class AbstractSimulationManager(metaclass=ABCMeta):
    """
    Generic simulation time manager.
    """

    @abstractmethod
    def get_iteration(self) -> SimulationIteration:
        """
        Get the current simulation iteration.
        :return: Get the current simulation current_simulation_state and time point
        """
        pass

    @abstractmethod
    def next_iteration(self) -> Optional[SimulationIteration]:
        """
        Move to the next iteration and return its simulation iteration.
        Returns None if we have reached the end of the simulation.
        """
        pass

    @abstractmethod
    def reached_end(self) -> bool:
        """
        Check if we have reached the end of the simulation.
        :return: Check whether simulation reached the end state.
        """
        pass

    @abstractmethod
    def number_of_iterations(self) -> int:
        """
        The number of iterations the simulation should be running for
        :return: Number of iterations of simulation.
        """
        pass
