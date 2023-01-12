from __future__ import annotations

import logging
import time

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.metric_callback import MetricCallback, run_metric_engine
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.abstract_runner import AbstractRunner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.simulation_log import SimulationLog

logger = logging.getLogger(__name__)


class MetricRunner(AbstractRunner):
    """Manager which executes metrics with multiple simulation logs."""

    def __init__(self, simulation_log: SimulationLog, metric_callback: MetricCallback) -> None:
        """
        Initialize the metric manager.
        :param simulation_log: A simulation log.
        :param metric_callback: A metric callback.
        """
        self._simulation_log = simulation_log
        self._metric_callback = metric_callback

    def run(self) -> RunnerReport:
        """
        Run through all metric runners with simulation logs.
        :return A list of runner reports.
        """
        start_time = time.perf_counter()

        # Initialize reports for all the simulations that will run
        report = RunnerReport(
            succeeded=True,
            error_message=None,
            start_time=start_time,
            end_time=None,
            planner_report=None,
            scenario_name=self._simulation_log.scenario.scenario_name,
            planner_name=self._simulation_log.planner.name(),
            log_name=self._simulation_log.scenario.log_name,
        )

        run_metric_engine(
            metric_engine=self._metric_callback.metric_engine,
            scenario=self._simulation_log.scenario,
            history=self._simulation_log.simulation_history,
            planner_name=self._simulation_log.planner.name(),
        )

        enc_time = time.perf_counter()

        # Only one metric runner, so it always updates the first report
        report.end_time = enc_time

        return report

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: Get the scenario.
        """
        return self._simulation_log.scenario

    @property
    def planner(self) -> AbstractPlanner:
        """
        :return: Get a planner.
        """
        return self._simulation_log.planner
