from typing import List, Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class LogFuturePlanner(AbstractPlanner):
    """
    Planner which just looks as future GT and returns it as a desired trajectory
    the input to this planner are detections.
    """

    # Inherited property, see superclass.
    requires_scenario: bool = True

    def __init__(self, scenario: AbstractScenario, num_poses: int, future_time_horizon: float):
        """Constructor of LogFuturePlanner."""
        self._scenario = scenario

        self._num_poses = num_poses
        self._future_time_horizon = future_time_horizon

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        pass

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_trajectory(self, current_input: List[PlannerInput]) -> List[AbstractTrajectory]:
        """Inherited, see superclass."""
        iteration = current_input[0].iteration
        current_state = self._scenario.get_ego_state_at_iteration(iteration.index)
        states = self._scenario.get_ego_future_trajectory(iteration.index, self._future_time_horizon, self._num_poses)
        return [InterpolatedTrajectory([current_state] + states)]
