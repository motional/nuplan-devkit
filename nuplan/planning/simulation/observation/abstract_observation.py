from abc import ABCMeta, abstractmethod
from typing import Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration


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
    def get_observation(self) -> Observation:
        """
        Get the current observation object.
        :return: Any type representing an observation, e.g., LidarPc, List[Box3D]
        """
        pass

    @abstractmethod
    def update_observation(self,
                           iteration: SimulationIteration,
                           next_iteration: SimulationIteration,
                           ego_state: EgoState) -> None:
        """
        Propagate observation into the next simulation iteration.
        Depending on the type of observation this may mean:
        1) Stepping to the next simulation iteration (point clouds).
        2) Running a planning model to compute agent trajectories and update their state accordingly.

        :param iteration: The current simulation iteration.
        :param next_iteration: the next simulation iteration that we update to.
        :param ego_state: The current ego state.
        """
        pass
