import abc

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class AbstractTracker(abc.ABC):
    """
    Interface for a generic tracker.
    """

    @abc.abstractmethod
    def track_trajectory(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        initial_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> DynamicCarState:
        """
        Return an ego state with updated dynamics according to the controller commands.
        :param current_iteration: The current simulation iteration.
        :param next_iteration: The desired next simulation iteration.
        :param initial_state: The current simulation iteration.
        :param trajectory: The reference trajectory to track.
        :return: The ego state to be propagated
        """
        pass
