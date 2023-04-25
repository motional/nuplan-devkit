from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from nuplan.common.utils.io_utils import save_object_as_pickle
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_file import MetricFile, MetricFileKey
from nuplan.planning.metrics.metric_result import MetricStatistics
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory

logger = logging.getLogger(__name__)


JSON_FILE_EXTENSION = ".pickle.temp"


def construct_dataframe(
    log_name: str, scenario_name: str, scenario_type: str, planner_name: str, metric_statistics: MetricStatistics
) -> Dict[str, Any]:
    """
    Construct a metric dataframe for metric results.
    :param log_name: A log name.
    :param scenario_name: Scenario name.
    :param scenario_type: Scenario type.
    :param planner_name: Planner name.
    :param metric_statistics: Metric statistics.
    :return A pandas dataframe for metric statistics.
    """
    # Construct all column header names.
    statistic_columns = {
        "log_name": log_name,
        "scenario_name": scenario_name,
        "scenario_type": scenario_type,
        "planner_name": planner_name,
        "metric_computator": metric_statistics.metric_computator,
        "metric_statistics_name": metric_statistics.name,
    }

    statistic_columns.update(metric_statistics.serialize_dataframe())
    return statistic_columns


class MetricsEngine:
    """The metrics engine aggregates and manages the instantiated metrics for a scenario."""

    def __init__(self, main_save_path: Path, metrics: Optional[List[AbstractMetricBuilder]] = None) -> None:
        """
        Initializer for MetricsEngine class
        :param metrics: Metric objects.
        """
        self._main_save_path = main_save_path
        if not is_s3_path(self._main_save_path):
            self._main_save_path.mkdir(parents=True, exist_ok=True)

        if metrics is None:
            self._metrics: List[AbstractMetricBuilder] = []
        else:
            self._metrics = metrics

    @property
    def metrics(self) -> List[AbstractMetricBuilder]:
        """Retrieve a list of metric results."""
        return self._metrics

    def add_metric(self, metric_builder: AbstractMetricBuilder) -> None:
        """TODO: Create the list of types needed from the history"""
        self._metrics.append(metric_builder)

    def write_to_files(self, metric_files: Dict[str, List[MetricFile]]) -> None:
        """
        Write to a file by constructing a dataframe
        :param metric_files: A dictionary of scenario names and a list of their metric files.
        """
        for scenario_name, metric_files in metric_files.items():  # type: ignore
            file_name = scenario_name + JSON_FILE_EXTENSION
            save_path = self._main_save_path / file_name
            dataframes = []
            for metric_file in metric_files:
                metric_file_key = metric_file.key  # type: ignore
                for metric_statistic in metric_file.metric_statistics:  # type: ignore
                    dataframe = construct_dataframe(
                        log_name=metric_file_key.log_name,
                        scenario_name=metric_file_key.scenario_name,
                        scenario_type=metric_file_key.scenario_type,
                        planner_name=metric_file_key.planner_name,
                        metric_statistics=metric_statistic,
                    )
                    dataframes.append(dataframe)

            if len(dataframes):
                save_object_as_pickle(save_path, dataframes)

    def compute_metric_results(
        self, history: SimulationHistory, scenario: AbstractScenario
    ) -> Dict[str, List[MetricStatistics]]:
        """
        Compute metrics in the engine
        :param history: History from simulation
        :param scenario: Scenario running this metric engine
        :return A list of metric statistics.
        """
        metric_results = {}
        for metric in self._metrics:
            try:
                start_time = time.perf_counter()
                metric_results[metric.name] = metric.compute(history, scenario=scenario)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.debug(f"Metric: {metric.name} running time: {elapsed_time:.2f} seconds.")
            except (NotImplementedError, Exception) as e:
                # Catch any error when computing a metric.
                logger.error(f"Running {metric.name} with error: {e}")
                raise RuntimeError(f"Metric Engine failed with: {e}")

        return metric_results

    def compute(
        self, history: SimulationHistory, scenario: AbstractScenario, planner_name: str
    ) -> Dict[str, List[MetricFile]]:
        """
        Compute metrics and return in a format of MetricStorageResult for each metric computation
        :param history: History from simulation
        :param scenario: Scenario running this metric engine
        :param planner_name: name of the planner
        :return A dictionary of scenario name and list of MetricStorageResult.
        """
        all_metrics_results = self.compute_metric_results(history=history, scenario=scenario)
        metric_files = defaultdict(list)
        for metric_name, metric_statistics_results in all_metrics_results.items():
            metric_file_key = MetricFileKey(
                metric_name=metric_name,
                log_name=scenario.log_name,
                scenario_name=scenario.scenario_name,
                scenario_type=scenario.scenario_type,
                planner_name=planner_name,
            )
            metric_file = MetricFile(key=metric_file_key, metric_statistics=metric_statistics_results)
            metric_file_name = scenario.scenario_type + "_" + scenario.scenario_name + "_" + planner_name
            metric_files[metric_file_name].append(metric_file)

        return metric_files
