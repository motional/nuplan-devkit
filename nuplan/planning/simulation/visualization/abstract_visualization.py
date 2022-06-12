from abc import ABCMeta, abstractmethod
from typing import Any, List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class AbstractVisualization(metaclass=ABCMeta):
    """
    Generic visualization interface.
    """

    @abstractmethod
    def render_scenario(self, scenario: AbstractScenario, render_goal: bool) -> None:
        """
        Render map.
        :param scenario: Scenario currently executed.
        :param render_goal: Whether to render the goal state.
        """
        pass

    @abstractmethod
    def render_ego_state(self, state_center: EgoState) -> None:
        """
        Render state of ego.
        :param state_center: State center.
        """
        pass

    @abstractmethod
    def render_polygon_trajectory(self, trajectory: List[StateSE2]) -> None:
        """
        Plot a trajectory as a polygon not as a trajectory.
        :param trajectory: trajectory to be plot.
        """
        pass

    @abstractmethod
    def render_trajectory(self, trajectory: List[StateSE2]) -> None:
        """
        Render trajectory.
        :param trajectory: Trajectory to be rendered.
        """
        pass

    @abstractmethod
    def render_observations(self, observations: Any) -> None:
        """
        Render observations
        :param observations: Either sensors or boxes.
        """
        pass

    @abstractmethod
    def render(self, iteration: SimulationIteration) -> None:
        """
        Trigger rendering of the snapshot.

        :param iteration: the current simulation iteration
        """
        pass
