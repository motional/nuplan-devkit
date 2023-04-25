from __future__ import annotations

import logging
import time
from typing import Any, Callable, List

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.abstract_runner import AbstractRunner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.simulation import Simulation

logger = logging.getLogger(__name__)


def for_each(fn: Callable[[Any], Any], items: List[Any]) -> None:
    """
    Call function on every item in items
    :param fn: function to be called fn(item)
    :param items: list of items
    """
    for item in items:
        fn(item)


class SimulationRunner(AbstractRunner):
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulation: Simulation, planner: AbstractPlanner):
        """
        Initialize the simulations manager
        :param simulation: Simulation which will be executed
        :param planner: to be used to compute the desired ego's trajectory
        """
        self._simulation = simulation
        self._planner = planner

    def _initialize(self) -> None:
        """
        Initialize the planner
        """
        # Execute specific callback
        self._simulation.callback.on_initialization_start(self._simulation.setup, self.planner)

        # Initialize Planner
        self.planner.initialize(self._simulation.initialize())

        # Execute specific callback
        self._simulation.callback.on_initialization_end(self._simulation.setup, self.planner)

    @property
    def planner(self) -> AbstractPlanner:
        """
        :return: Planner used by the SimulationRunner
        """
        return self._planner

    @property
    def simulation(self) -> Simulation:
        """
        :return: Simulation used by the SimulationRunner
        """
        return self._simulation

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: Get the scenario relative to the simulation.
        """
        return self.simulation.scenario

    def run(self) -> RunnerReport:
        """
        Run through all simulations. The steps of execution follow:
         - Initialize all planners
         - Step through simulations until there no running simulation
        :return: List of SimulationReports containing the results of each simulation
        """
        start_time = time.perf_counter()

        # Initialize reports for all the simulations that will run
        report = RunnerReport(
            succeeded=True,
            error_message=None,
            start_time=start_time,
            end_time=None,
            planner_report=None,
            scenario_name=self._simulation.scenario.scenario_name,
            planner_name=self.planner.name(),
            log_name=self._simulation.scenario.log_name,
        )

        # Execute specific callback
        self.simulation.callback.on_simulation_start(self.simulation.setup)

        # Initialize all simulations
        self._initialize()

        while self.simulation.is_simulation_running():
            # Execute specific callback
            self.simulation.callback.on_step_start(self.simulation.setup, self.planner)

            # Perform step
            planner_input = self._simulation.get_planner_input()
            logger.debug("Simulation iterations: %s" % planner_input.iteration.index)

            # Execute specific callback
            self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)

            # Plan path based on all planner's inputs
            trajectory = self.planner.compute_trajectory(planner_input)

            # Propagate simulation based on planner trajectory
            self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)
            self.simulation.propagate(trajectory)

            # Execute specific callback
            self.simulation.callback.on_step_end(self.simulation.setup, self.planner, self.simulation.history.last())

            # Store reports for simulations which just finished running
            current_time = time.perf_counter()
            if not self.simulation.is_simulation_running():
                report.end_time = current_time

        # Execute specific callback
        self.simulation.callback.on_simulation_end(self.simulation.setup, self.planner, self.simulation.history)

        planner_report = self.planner.generate_planner_report()
        report.planner_report = planner_report

        return report
