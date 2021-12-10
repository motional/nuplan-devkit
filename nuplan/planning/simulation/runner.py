import logging
import pathlib
import pickle
import traceback
from dataclasses import dataclass
from typing import List, Optional, Union

from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool

logger = logging.getLogger(__name__)


@dataclass
class SimulationReport:
    succeeded: bool  # True if simulation was successful
    error_message: Optional[str]  # None if simulation succeeded, traceback if it failed


def run_simulation(simulation: Simulation,
                   exit_on_failure: bool = False,
                   thread_idx: Optional[int] = None) -> SimulationReport:
    """
    Proxy for calling simulation.
    :param simulation: A simulation
    :param thread_idx: Thread index used for this simulation
    :param exit_on_failure: If true, raises an exception when the simulation fails
    :return report for the simulation
    """

    try:
        simulation.run(thread_idx)
        return SimulationReport(True, None)
    except Exception as e:
        error = traceback.format_exc()

        # Print to the terminal
        logger.warning("----------- Simulation failed: with the following trace:")
        traceback.print_exc()
        logger.warning(
            f"Simulation token: {simulation.scenario.token}, name: {simulation.scenario.scenario_name} "
            f"failed with error:\n {e}")
        logger.warning("----------- Simulation failed!")

        # Fail if desired
        if exit_on_failure:
            raise RuntimeError('Simulation failed')

        return SimulationReport(False, error)


def run_planner_through_scenarios(simulations: List[Simulation],
                                  worker: WorkerPool,
                                  num_gpus: Optional[Union[int, float]],
                                  num_cpus: Optional[int],
                                  exit_on_failure: bool = False) -> None:
    """
    Execute multiple simulations
    :param simulations: A list of simulations.
    :param worker: for submitting tasks
    :param num_gpus: if None, no GPU will be used, otherwise number (also fractional) of GPU used per simulation
    :param num_cpus: if None, all available CPU threads are used, otherwise number of threads used
    :param exit_on_failure: If true, raises an exception when the simulation fails
    """

    # Start simulations
    number_of_sims = len(simulations)
    logger.info(f"Starting {number_of_sims} simulations using {worker.__class__.__name__}!")
    results: List[SimulationReport] = worker.map(Task(fn=run_simulation, num_gpus=num_gpus, num_cpus=num_cpus),
                                                 simulations, exit_on_failure)

    # Notify user about the result of simulations
    number_of_successful = len([result for result in results if result.succeeded])
    logger.info(f"Number of successful simulations: {number_of_successful}")
    logger.info(f"Number of failed simulations: {number_of_sims - number_of_successful}")

    # Print out all failed simulation unique identifier
    for result, simulation in zip(results, simulations):
        if not result.succeeded:
            logger.warning(f"Failed Simulation: {simulation.identifier}.\n{result.error_message}")
