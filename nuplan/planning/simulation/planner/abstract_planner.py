import abc
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Type, Union

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.planner.planner_report import PlannerReport
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


@dataclass(frozen=True)
class PlannerInitialization:
    """
    This class represents required data to initialize a planner.
    """

    route_roadblock_ids: List[str]  # Roadblock ids comprising goal route
    mission_goal: StateSE2  # The mission goal which commonly is not achievable in a single scenario
    map_api: AbstractMap  # The API towards maps.


@dataclass(frozen=True)
class PlannerInput:
    """
    Input to a planner for which a trajectory should be computed.
    """

    iteration: SimulationIteration  # Iteration and time in a simulation progress
    history: SimulationHistoryBuffer  # Rolling buffer containing past observations and states.
    traffic_light_data: Optional[List[TrafficLightStatusData]] = None  # The traffic light status data


class AbstractPlanner(abc.ABC):
    """
    Interface for a generic ego vehicle planner.
    """

    # Whether this planner can consume multiple scenarios at once during inference.
    # If this is false, only one simulation scenario will be run at a time.
    consume_batched_inputs: bool = False

    # Whether the planner requires the scenario object to be passed at construction time.
    # This can be set to true only for oracle planners and cannot be used for submissions.
    requires_scenario: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> Type['AbstractPlanner']:  # type: ignore
        """
        Define attributes needed by all planners, take care when overriding.
        :param cls: class being constructed.
        :param args: arguments to constructor.
        :param kwargs: keyword arguments to constructor.
        """
        instance: Type['AbstractPlanner'] = super().__new__(cls)  # type: ignore
        instance._compute_trajectory_runtimes = []
        return instance

    @abstractmethod
    def name(self) -> str:
        """
        :return string describing name of this planner.
        """
        pass

    @abc.abstractmethod
    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """
        Initialize planner
        :param initialization: List of initialization classes.
            This is a list only on case consume_batched_inputs is True, otherwise it has a single entry in list
            In this case the list represents batched simulations.
        """
        pass

    @abc.abstractmethod
    def observation_type(self) -> Type[Observation]:
        """
        :return Type of observation that is expected in compute_trajectory.
        """
        pass

    @abc.abstractmethod
    def compute_planner_trajectory(self, current_input: List[PlannerInput]) -> List[AbstractTrajectory]:
        """
        Computes the ego vehicle trajectory.
        :param current_input: List of planner inputs for where for each of them trajectory should be computed
            In this case the list represents batched simulations. In case consume_batched_inputs is False
            the list has only single element
        :return: Trajectories representing the predicted ego's position in future for every input iteration
            In case consume_batched_inputs is False, return only a single trajectory in a list.
        """
        pass

    def compute_single_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Compute trajectory only for a single planner input
        :param current_input: input to the planner
        :return: Trajectory representing the predicted ego's position in future for the input iteration
        """
        return self.compute_trajectory([current_input])[0]

    def compute_trajectory(self, current_input: List[PlannerInput]) -> List[AbstractTrajectory]:
        """
        Computes the ego vehicle trajectory, where we check that if planner can not consume batched inputs,
            we require that the input list has exactly one element
        :param current_input: List of planner inputs for where for each of them trajectory should be computed
            In this case the list represents batched simulations. In case consume_batched_inputs is False
            the list has only single element
        :return: Trajectories representing the predicted ego's position in future for every input iteration
            In case consume_batched_inputs is False, return only a single trajectory in a list.
        """
        self.validate_inputs(current_input)
        start_time = time.perf_counter()
        # If it raises an exception, catch to record the time then re-raise it.
        try:
            trajectory = self.compute_planner_trajectory(current_input)
        except Exception as e:
            self._compute_trajectory_runtimes.append(time.perf_counter() - start_time)
            raise e

        self._compute_trajectory_runtimes.append(time.perf_counter() - start_time)
        return trajectory

    def initialize_with_check(self, initialization: List[PlannerInitialization]) -> None:
        """
        Initialize planner where we check that if planner can not consume batched inputs, we require that the input
            list has exactly one element
        :param initialization: List of initialization classes.
            This is a list only in case consume_batched_inputs is True, otherwise it has a single entry in list
            In this case the list represents batched simulations.
        """
        self.validate_inputs(initialization)
        return self.initialize(initialization)

    def validate_inputs(self, data: Union[List[PlannerInput], List[PlannerInitialization]]) -> None:
        """
        Validate that the size of the input data correspond the the fact whether this planner can consume batched inputs
            This function will raise in case length of data and consume_batched_inputs does not match.
        :param data: input data.
        """
        if len(data) == 0:
            raise RuntimeError("The inputs to the planner can not be an empty list!")
        if not self.consume_batched_inputs and len(data) > 1:
            raise RuntimeError(
                f"Planner: {self.name()} can not consume batched inputs, but {len(data)} inputs was provided!"
            )

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """
        Generate a report containing runtime stats from the planner.
        By default, returns a report containing the time-series of compute_trajectory runtimes.
        :param clear_stats: whether or not to clear stored stats after creating report.
        :return: report containing planner runtime stats.
        """
        report = PlannerReport(compute_trajectory_runtimes=self._compute_trajectory_runtimes)
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
        return report
