from typing import Type

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import Observation, Sensors
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory


class EndToEndPlanner(AbstractPlanner):
    """
    Planner which just looks as future GT and returns it as a desired trajectory the input to this planner are sensors.

    NOTE: this planner does not really consumes sensor data. It does just illustrate how an end-to-end planner
        could be implemented. You will have to implement the logic yourself!
    """

    def __init__(self,
                 horizon_seconds: float,
                 sampling_time: float,
                 acceleration: npt.NDArray[np.float32]):
        self.planner = SimplePlanner(horizon_seconds=horizon_seconds,
                                     sampling_time=sampling_time,
                                     acceleration=acceleration)

    def initialize(self,
                   expert_goal_state: StateSE2,
                   mission_goal: StateSE2,
                   map_name: str,
                   map_api: AbstractMap) -> None:
        """ Inherited, see superclass. """
        pass

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """ Inherited, see superclass. """
        return Sensors  # type: ignore

    def compute_trajectory(self, iteration: SimulationIteration,
                           history: SimulationHistoryBuffer) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """
        return self.planner.compute_trajectory(iteration, history)
