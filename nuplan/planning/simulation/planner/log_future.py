from typing import Type

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import Detections, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory


class LogFuturePlanner(AbstractPlanner):
    """
    Planner which just looks as future GT and returns it as a desired trajectory
    the input to this planner are detections.
    """

    def __init__(self, scenario: AbstractScenario, number_of_next_samples: int = 10):
        self.scenario = scenario
        self.number_of_next_samples = number_of_next_samples

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
        return Detections  # type: ignore

    def compute_trajectory(self, iteration: SimulationIteration,
                           history: SimulationHistoryBuffer) -> AbstractTrajectory:
        """ Inherited, see superclass. """
        states = self.scenario.get_ego_trajectory_slice(iteration.index, iteration.index + self.number_of_next_samples)
        return InterpolatedTrajectory(states)
