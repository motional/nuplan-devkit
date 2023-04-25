from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

logger = logging.getLogger(__name__)


class Simulation:
    """
    This class queries data for initialization of a planner, and propagates simulation a step forward based on the
        planned trajectory of a planner.
    """

    def __init__(
        self,
        simulation_setup: SimulationSetup,
        callback: Optional[AbstractCallback] = None,
        simulation_history_buffer_duration: float = 2,
    ):
        """
        Create Simulation.
        :param simulation_setup: Configuration that describes the simulation.
        :param callback: A callback to be executed for this simulation setup
        :param simulation_history_buffer_duration: [s] Duration to pre-load scenario into the buffer.
        """
        if simulation_history_buffer_duration < simulation_setup.scenario.database_interval:
            raise ValueError(
                f"simulation_history_buffer_duration {simulation_history_buffer_duration} has to be larger than the scenario database_interval {simulation_setup.scenario.database_interval}"
            )

        # Store all engines
        self._setup = simulation_setup

        # Proxy
        self._time_controller = simulation_setup.time_controller
        self._ego_controller = simulation_setup.ego_controller
        self._observations = simulation_setup.observations
        self._scenario = simulation_setup.scenario
        self._callback = MultiCallback([]) if callback is None else callback

        # History where the steps of a simulation are stored
        self._history = SimulationHistory(self._scenario.map_api, self._scenario.get_mission_goal())

        # Rolling window of past states
        # We add self._scenario.database_interval to the buffer duration here to ensure that the minimum
        # simulation_history_buffer_duration is satisfied
        self._simulation_history_buffer_duration = simulation_history_buffer_duration + self._scenario.database_interval

        # The + 1 here is to account for duration. For example, 20 steps at 0.1s starting at 0s will have a duration
        # of 1.9s. At 21 steps the duration will achieve the target 2s duration.
        self._history_buffer_size = int(self._simulation_history_buffer_duration / self._scenario.database_interval) + 1
        self._history_buffer: Optional[SimulationHistoryBuffer] = None

        # Flag that keeps track whether simulation is still running
        self._is_simulation_running = True

    def __reduce__(self) -> Tuple[Type[Simulation], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._setup, self._callback, self._simulation_history_buffer_duration)

    def is_simulation_running(self) -> bool:
        """
        Check whether a simulation reached the end
        :return True if simulation hasn't reached the end, otherwise false.
        """
        return not self._time_controller.reached_end() and self._is_simulation_running

    def reset(self) -> None:
        """
        Reset all internal states of simulation.
        """
        # Clear created log
        self._history.reset()

        # Reset all simulation internal members
        self._setup.reset()

        # Clear history buffer
        self._history_buffer = None

        # Restart simulation
        self._is_simulation_running = True

    def initialize(self) -> PlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
        :return data needed for planner initialization.
        """
        self.reset()

        # Initialize history from scenario
        self._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
            self._history_buffer_size, self._scenario, self._observations.observation_type()
        )

        # Initialize observations
        self._observations.initialize()

        # Add the current state into the history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

        # Return the planner initialization structure for this simulation
        return PlannerInitialization(
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            mission_goal=self._scenario.get_mission_goal(),
            map_api=self._scenario.map_api,
        )

    def get_planner_input(self) -> PlannerInput:
        """
        Construct inputs to the planner for the current iteration step
        :return Inputs to the planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, stepping can not be performed!")

        # Extract current state
        iteration = self._time_controller.get_iteration()

        # Extract traffic light status data
        traffic_light_data = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))
        logger.debug(f"Executing {iteration.index}!")
        return PlannerInput(iteration=iteration, history=self._history_buffer, traffic_light_data=traffic_light_data)

    def propagate(self, trajectory: AbstractTrajectory) -> None:
        """
        Propagate the simulation based on planner's trajectory and the inputs to the planner
        This function also decides whether simulation should still continue. This flag can be queried through
        reached_end() function
        :param trajectory: computed trajectory from planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, simulation can not be propagated!")

        # Measurements
        iteration = self._time_controller.get_iteration()
        ego_state, observation = self._history_buffer.current_state
        traffic_light_status = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))

        # Add new sample to history
        logger.debug(f"Adding to history: {iteration.index}")
        self._history.add_sample(
            SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status)
        )

        # Propagate state to next iteration
        next_iteration = self._time_controller.next_iteration()

        # Propagate state
        if next_iteration:
            self._ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
            self._observations.update_observation(iteration, next_iteration, self._history_buffer)
        else:
            self._is_simulation_running = False

        # Append new state into history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: used scenario in this simulation.
        """
        return self._scenario

    @property
    def setup(self) -> SimulationSetup:
        """
        :return: Setup for this simulation.
        """
        return self._setup

    @property
    def callback(self) -> AbstractCallback:
        """
        :return: Callback for this simulation.
        """
        return self._callback

    @property
    def history(self) -> SimulationHistory:
        """
        :return History from the simulation.
        """
        return self._history

    @property
    def history_buffer(self) -> SimulationHistoryBuffer:
        """
        :return SimulationHistoryBuffer from the simulation.
        """
        if self._history_buffer is None:
            raise RuntimeError(
                "_history_buffer is None. Please initialize the buffer by calling Simulation.initialize()"
            )
        return self._history_buffer
