import concurrent.futures
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union

from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.runner.abstract_runner import AbstractRunner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool

logger = logging.getLogger(__name__)


def run_simulation(sim_runner: AbstractRunner, exit_on_failure: bool = False) -> RunnerReport:
    """
    Proxy for calling simulation.
    :param sim_runner: A simulation runner which will execute all batched simulations.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    :return report for the simulation.
    """
    # Store start time so that if the simulations fail, we know how long they ran for
    start_time = time.perf_counter()
    try:
        return sim_runner.run()
    except Exception as e:
        error = traceback.format_exc()

        # Print to the terminal
        logger.warning("----------- Simulation failed: with the following trace:")
        traceback.print_exc()
        logger.warning(f"Simulation failed with error:\n {e}")

        # Log the failed scenario log/tokens
        failed_scenarios = f"[{sim_runner.scenario.log_name}, {sim_runner.scenario.scenario_name}]\n"
        logger.warning(f"\nFailed simulation [log,token]:\n {failed_scenarios}")

        logger.warning("----------- Simulation failed!")

        # Fail if desired
        if exit_on_failure:
            raise RuntimeError('Simulation failed')

        end_time = time.perf_counter()
        report = RunnerReport(
            succeeded=False,
            error_message=error,
            start_time=start_time,
            end_time=end_time,
            planner_report=None,
            scenario_name=sim_runner.scenario.scenario_name,
            planner_name=sim_runner.planner.name(),
            log_name=sim_runner.scenario.log_name,
        )

        return report


def execute_runners(
    runners: List[AbstractRunner],
    worker: WorkerPool,
    num_gpus: Optional[Union[int, float]],
    num_cpus: Optional[int],
    exit_on_failure: bool = False,
    verbose: bool = False,
) -> List[RunnerReport]:
    """
    Execute multiple simulation runners or metric runners.
    :param runners: A list of simulations to be run.
    :param worker: for submitting tasks.
    :param num_gpus: if None, no GPU will be used, otherwise number (also fractional) of GPU used per simulation.
    :param num_cpus: if None, all available CPU threads are used, otherwise number of threads used.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    """
    # Validating
    assert len(runners) > 0, 'No scenarios found to simulate!'

    # Start simulations
    number_of_sims = len(runners)
    logger.info(f"Starting {number_of_sims} simulations using {worker.__class__.__name__}!")
    reports: List[RunnerReport] = worker.map(
        Task(fn=run_simulation, num_gpus=num_gpus, num_cpus=num_cpus), runners, exit_on_failure, verbose=verbose
    )
    # Store the results in a dictionary so we can easily store error tracebacks in the next step, if needed
    results: Dict[Tuple[str, str, str], RunnerReport] = {
        (report.scenario_name, report.planner_name, report.log_name): report for report in reports
    }

    # Iterate over runners, finding the callbacks which may have run asynchronously, and gathering their results
    simulations_runners = (runner for runner in runners if isinstance(runner, SimulationRunner))
    relevant_simulations = ((runner.simulation, runner) for runner in simulations_runners)
    callback_futures_lists = (
        (callback.futures, simulation, runner)
        for (simulation, runner) in relevant_simulations
        for callback in simulation.callback.callbacks
        if isinstance(callback, MetricCallback) or isinstance(callback, SimulationLogCallback)
    )
    callback_futures_map = {
        future: (simulation.scenario.scenario_name, runner.planner.name(), simulation.scenario.log_name)
        for (futures, simulation, runner) in callback_futures_lists
        for future in futures
    }
    for future in concurrent.futures.as_completed(callback_futures_map.keys()):
        try:
            future.result()
        except Exception:
            error_message = traceback.format_exc()
            runner_report = results[callback_futures_map[future]]
            runner_report.error_message = error_message
            runner_report.succeeded = False
            runner_report.end_time = time.perf_counter()

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0
    runner_reports: List[RunnerReport] = list(results.values())
    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            logger.warning("Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    number_of_failures = number_of_sims - number_of_successful
    logger.info(f"Number of successful simulations: {number_of_successful}")
    logger.info(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        logger.info(f"Failed simulations [log, token]:\n{failed_simulations}")

    return runner_reports
