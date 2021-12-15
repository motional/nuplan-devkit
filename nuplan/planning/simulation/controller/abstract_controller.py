from abc import ABCMeta, abstractmethod

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory


class AbstractEgoController(metaclass=ABCMeta):
    """
    Interface for generic eo controllers.
    """

    @abstractmethod
    def get_state(self) -> EgoState:
        """
        Returns the current ego state.
        :return: The current ego state.
        """
        pass

    @abstractmethod
    def update_state(self,
                     current_iteration: SimulationIteration,
                     next_iteration: SimulationIteration,
                     ego_state: EgoState,
                     trajectory: AbstractTrajectory) -> None:
        """
        Update ego's state from current iteration to next iteration.

        :param current_iteration: The current simulation iteration.
        :param next_iteration: The desired next simulation iteration.
        :param ego_state: The current ego state.
        :param trajectory: The output trajectory of a planner.
        """
        pass
