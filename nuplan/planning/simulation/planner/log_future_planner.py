import itertools
import logging
from typing import List, Optional, Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

logger = logging.getLogger(__name__)


class LogFuturePlanner(AbstractPlanner):
    """
    Planner which just looks as future GT and returns it as a desired trajectory
    the input to this planner are detections.
    """

    # Inherited property, see superclass.
    requires_scenario: bool = True

    def __init__(self, scenario: AbstractScenario, num_poses: int, future_time_horizon: float):
        """
        Constructor of LogFuturePlanner.
        :param scenario: The scenario the planner is running on.
        :param num_poses: The number of poses to plan for.
        :param future_time_horizon: [s] The horizon length to plan for.
        """
        self._scenario = scenario

        self._num_poses = num_poses
        self._future_time_horizon = future_time_horizon
        self._trajectory: Optional[AbstractTrajectory] = None

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        pass

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        current_state = self._scenario.get_ego_state_at_iteration(current_input.iteration.index)
        try:
            states = self._scenario.get_ego_future_trajectory(
                current_input.iteration.index, self._future_time_horizon, self._num_poses
            )
            self._trajectory = InterpolatedTrajectory(list(itertools.chain([current_state], states)))
        except AssertionError:
            logger.warning("Cannot retrieve future ego trajectory. Using previous computed trajectory.")
            if self._trajectory is None:
                raise RuntimeError("Future ego trajectory cannot be retrieved from the scenario!")

        return self._trajectory
