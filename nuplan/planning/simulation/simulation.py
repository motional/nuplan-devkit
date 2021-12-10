from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Type, cast

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup, validate_planner_setup
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Simulation:
    """
    Agent-based planning simulation.
    """

    def __init__(self,
                 simulation_setup: SimulationSetup, planner: AbstractPlanner,
                 callbacks: Optional[List[AbstractCallback]] = None,
                 enable_progress_bar: bool = True,
                 simulation_history_buffer_duration: float = 2):
        """
        Create Simulation.
        :param simulation_setup: Configuration that describes the simulation.
        :param planner: Planner to be used for this simulation
        :param enable_progress_bar: If true, progress bar will be used to show progress along simulation scenario
        :param simulation_history_buffer_duration: [s] Duration to pre-load scenario into the buffer
        """
        # Validate the setup
        validate_planner_setup(simulation_setup, planner)

        # Store all engines
        self._setup = simulation_setup
        self._planner = planner

        # Proxy
        self._simulation_manager = simulation_setup.simulation_manager
        self._ego_controller = simulation_setup.ego_controller
        self._observations = simulation_setup.observations
        self._scenario = simulation_setup.scenario

        # Callbacks
        self._callbacks = [] if callbacks is None else callbacks

        # History where the steps of a simulation are stored
        self._history = SimulationHistory(self._scenario.map_api, self._scenario.get_mission_goal())

        # Rolling window of past states
        buffer_size = int(simulation_history_buffer_duration / self._scenario.database_interval + 1)
        self._history_buffer = SimulationHistoryBuffer(buffer_size)

        # Config
        self._enable_progress_bar = enable_progress_bar
        self._simulation_history_buffer_duration = simulation_history_buffer_duration

    def __reduce__(self) -> Tuple[Type[Simulation], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._setup, self._planner, self._callbacks,
                                 self._enable_progress_bar, self._simulation_history_buffer_duration,))

    @property
    def identifier(self) -> str:
        """
        :return: identifier of the simulation used as file name
        """
        return f"simulation" \
               f"_{self._scenario.scenario_name}" \
               f"_{self._planner.name()}" \
               f"_{self._observations.__class__.__name__}" \
               f"_{self._ego_controller.__class__.__name__}"

    @property
    def history(self) -> SimulationHistory:
        """
        :return history from the simulation
        """
        return self._history

    def initialize(self) -> None:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
         - Initialize Visualization with scenario
        """
        # Execute all callbacks
        for callback in self._callbacks:
            callback.on_initialization_start(self._setup, self._planner)

        # Initialize planner
        self._planner.initialize(self._scenario.get_expert_goal_state(), self._scenario.get_mission_goal(),
                                 self._scenario.map_api.map_name, self._scenario.map_api)

        # Initialize queues
        self._history_buffer.initialize(self._scenario, self._observations.observation_type())

        # Execute all callbacks
        for callback in self._callbacks:
            callback.on_initialization_end(self._setup, self._planner)

    def reached_end(self) -> bool:
        """
        Check whether a simulation reached the end
        :return True if simulation reached the end, otherwise false
        """
        return cast(bool, self._simulation_manager.reached_end())

    def step(self) -> bool:
        """
        Perform a simulation step

        :return True if simulation should continue, False if it should be terminated
        """
        # Execute all callbacks
        for callback in self._callbacks:
            callback.on_step_start(self._setup, self._planner)

        # Extract current state
        iteration = self._simulation_manager.get_iteration()
        logger.debug(f"Executing {iteration.index}!")

        # Measurements
        ego_state = self._ego_controller.get_state()
        observation = self._observations.get_observation()

        # Add new measurements to buffer
        self._history_buffer.append(ego_state, observation)

        # Compute trajectory
        for callback in self._callbacks:
            callback.on_planner_start(self._setup, self._planner)

        trajectory = self._planner.compute_trajectory(iteration, self._history_buffer)
        if not trajectory:
            raise RuntimeError(
                f"Planner {self._planner.name()} did not result in a trajectory in iteration {iteration.index} "
                f"on scenario {self._scenario.scenario_name}")

        for callback in self._callbacks:
            callback.on_planner_end(self._setup, self._planner, trajectory)

        # Add new sample to history
        sample = SimulationHistorySample(iteration, ego_state, trajectory, observation)
        self._history.add_sample(sample)

        # Propagate state to next iteration
        next_iteration = self._simulation_manager.next_iteration()
        if next_iteration is None:
            # Execute all callbacks
            for callback in self._callbacks:
                callback.on_simulation_manager_end(self._setup, self._planner, self._history)
            # Execute all callbacks
            for callback in self._callbacks:
                callback.on_step_end(self._setup, self._planner, sample)
            return False

        # Propagate state
        self._ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
        self._observations.update_observation(iteration, next_iteration, ego_state)

        # Execute all callbacks
        for callback in self._callbacks:
            callback.on_step_end(self._setup, self._planner, sample)

        return True

    def run(self, thread_idx: Optional[int] = None) -> SimulationHistory:
        """
        Run the entire simulation scenario.
        """
        logger.info(f"Starting simulation with scenario name: {self._scenario.scenario_name}!")

        # Execute all callbacks
        for callback in self._callbacks:
            callback.on_simulation_start(self._setup)

        # Initialize simulation
        self.initialize()

        # Progress Bar
        progress = None
        if self._enable_progress_bar:
            desc = f'Scenario {self._scenario.scenario_name} ({self._scenario.scenario_type})'
            progress = tqdm(total=self._simulation_manager.number_of_iterations(), position=thread_idx, desc=desc,
                            leave=False)
            progress.update(1)

        while not self.reached_end():
            # Perform simulation step
            should_continue = self.step()

            # Update progress bar
            if progress:
                progress.update(1)

            # Terminate in case it was requested
            if not should_continue:
                break

        # Execute all callbacks
        for callback in self._callbacks:
            callback.on_simulation_end(self._setup, self._planner, self._history)

        # Log the simulation end
        logger.info(f"Finished simulation with scenario name: {self._scenario.scenario_name}!")

        return self._history

    @property
    def planner(self) -> AbstractPlanner:
        """
        :return: used planner in this simulation
        """
        return self._planner

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: used scenario in this simulation
        """
        return self._scenario
