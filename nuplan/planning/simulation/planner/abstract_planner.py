from abc import ABCMeta, abstractmethod
from typing import Type

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory


class AbstractPlanner(metaclass=ABCMeta):
    """
    Interface for a generic ego vehicle planner.
    """

    @abstractmethod
    def initialize(self,
                   expert_goal_state: StateSE2,
                   mission_goal: StateSE2,
                   map_name: str,
                   map_api: AbstractMap) -> None:
        """
        Initialize planner

        :param expert_goal_state: desired state which was achieved by an expert within an scenario
        :param mission_goal: mission goal far along desired long-term route, not achievable within scenario length
        :param map_name: name of a map used for the scenario
        :param map_api: abstract map api for accessing the maps
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Name of a planner

        :return string describing name of this planner
        """
        pass

    @abstractmethod
    def observation_type(self) -> Type[Observation]:
        """
        Type of observation that is expected in compute_trajectory.
        """
        pass

    @abstractmethod
    def compute_trajectory(self, iteration: SimulationIteration,
                           history: SimulationHistoryBuffer) -> AbstractTrajectory:
        """
        Computes the ego vehicle trajectory.
        :param iteration: Current iteration of a simulation for which trajectory should be computed.
        :param history: Past simulation states including the state at the current time step [t_-N, ..., t_-1, t_0]
                        The buffer contains the past ego trajectory and past observations.
        :return: Trajectory representing the desired ego's position in future.
        """
        pass
