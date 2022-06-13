from abc import ABCMeta, abstractmethod
from typing import Type

from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class AbstractObservation(metaclass=ABCMeta):
    """
    Interface for a generic observation engine.
    Observations can include sensor data (pointclouds, images, velocities), tracker ouputs (bounding boxes) and more.
    """

    def observation_type(self) -> Type[Observation]:
        """
        Returns the type of observation.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the observation (all internal states should be reseted, if any).
        """
        pass

    @abstractmethod
    def get_observation(self) -> Observation:
        """
        Get the current observation object.
        :return: Any type representing an observation, e.g., LidarPc, TrackedObjects
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize observation if needed.
        """
        pass

    @abstractmethod
    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """
        Propagate observation into the next simulation iteration.
        Depending on the type of observation this may mean:
        1) Stepping to the next simulation iteration (point clouds).
        2) Running a planning model to compute agent trajectories and update their state accordingly.

        :param iteration: The current simulation iteration.
        :param next_iteration: the next simulation iteration that we update to.
        :param history: Past simulation states including the state at the current time step [t_-N, ..., t_-1, t_0]
                        The buffer contains the past ego trajectory and past observations.
        """
        pass
