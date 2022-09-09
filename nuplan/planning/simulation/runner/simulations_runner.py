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


class SimulationsRunner(AbstractRunner):
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulations: List[Simulation], planner: AbstractPlanner):
        """
        Initialize the simulations manager
        :param simulations: List of simulations which will be executed
        :param planner: which should be used to compute the desired ego's trajectory
        """
        self._simulations = simulations
        self._planner = planner

        if len(simulations) > 1 and not planner.consume_batched_inputs:
            # Raise in case planner is not able to consume batched inputs and it can consume only single input
            raise RuntimeError(
                f"Planner: {planner.name()} can not consume batches inputs from batches: {len(simulations)}!"
            )

    def get_running_simulations(self) -> List[Simulation]:
        """
        :return: List of simulations that are still running
        """
        return [sim for sim in self.simulations if sim.is_simulation_running()]

    def _initialize(self) -> None:
        """
        Initialize the planner
        """
        # Execute specific callback
        for_each(lambda sim: sim.callback.on_initialization_start(sim.setup, self.planner), self.simulations)

        # Initialize Planner
        self.planner.initialize_with_check([simulation.initialize() for simulation in self.simulations])

        # Execute specific callback
        for_each(lambda sim: sim.callback.on_initialization_end(sim.setup, self.planner), self.simulations)

    @property
    def simulations(self) -> List[Simulation]:
        """
        :return: List of simulations run by the SimulationRunner
        """
        return self._simulations

    @property
    def scenarios(self) -> List[AbstractScenario]:
        """
        :return: Get a list of scenarios.
        """
        return [sim.scenario for sim in self.simulations]

    @property
    def planner(self) -> AbstractPlanner:
        """
        :return: Planner used by the SimulationRunner
        """
        return self._planner

    def run(self) -> List[RunnerReport]:
        """
        Run through all simulations. The steps of execution follow:
         - Initialize all planners
         - Step through simulations until there no running simulation
        :return: List of SimulationReports containing the results of each simulation
        """
        start_time = time.perf_counter()

        # Initialize reports for all the simulations that will run
        reports = []
        for simulation in self.simulations:
            reports.append(
                RunnerReport(
                    succeeded=True,
                    error_message=None,
                    start_time=start_time,
                    end_time=None,
                    planner_report=None,
                    scenario_name=simulation.scenario.scenario_name,
                    planner_name=self.planner.name(),
                    log_name=simulation.scenario.log_name,
                )
            )

        # Execute specific callback
        for_each(lambda sim: sim.callback.on_simulation_start(sim.setup), self.simulations)

        # Initialize all simulations
        self._initialize()

        while len(simulations := self.get_running_simulations()) > 0:
            # Extract all running simulations
            logger.debug(f"Number of running simulations: {len(simulations)}")

            # Execute specific callback
            for_each(lambda sim: sim.callback.on_step_start(sim.setup, self.planner), simulations)

            # Perform step
            planner_inputs = [simulation.get_planner_input() for simulation in simulations]
            logger.debug(
                f"Simulation iterations: {[planner_input.iteration.index for planner_input in planner_inputs]}"
            )

            # Execute specific callback
            for_each(lambda sim: sim.callback.on_planner_start(sim.setup, self.planner), simulations)

            # Plan path based on all planner's inputs
            trajectories = self.planner.compute_trajectory(planner_inputs)
            if len(trajectories) != len(planner_inputs):
                # Raise in case the planner did not return the right number of output trajectories
                raise RuntimeError(
                    "The length of planner input and output is not "
                    f"the same {len(trajectories)} != {len(planner_inputs)}!"
                )

            # Propagate all simulations based on planner trajectory
            for trajectory, simulation in zip(trajectories, simulations):
                simulation.callback.on_planner_end(simulation.setup, self.planner, trajectory)
                simulation.propagate(trajectory)

            # Execute specific callback
            for_each(
                lambda sim: sim.callback.on_step_end(simulation.setup, self.planner, simulation.history.last()),
                simulations,
            )

            # Store reports for simulations which just finished running
            current_time = time.perf_counter()
            for simulation in simulations:
                if not simulation._is_simulation_running:
                    sim_index = self.simulations.index(simulation)
                    reports[sim_index].end_time = current_time

        # Execute specific callback
        for_each(lambda sim: sim.callback.on_simulation_end(sim.setup, self.planner, sim.history), self.simulations)

        planner_report = self.planner.generate_planner_report()
        for report in reports:
            report.planner_report = planner_report

        return reports
