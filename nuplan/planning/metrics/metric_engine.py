from __future__ import annotations

import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_file import MetricFile, MetricFileKey
from nuplan.planning.metrics.metric_result import MetricStatistics
from nuplan.planning.simulation.history.simulation_history import SimulationHistory

logger = logging.getLogger(__name__)


class MetricsEngine:

    def __init__(self,
                 main_save_path: Path,
                 timestamp: int,
                 scenario_type: str,
                 metrics: Optional[List[AbstractMetricBuilder]] = None):
        """
        The metrics engine aggregates and manages the instantiated metrics for a scenario.
        :param main_save_path: Main saving path.
        :param timestamp: Simulation timestamp.
        :param scenario_type: Scenario type.
        :param metrics: Metric objects.
        """

        self._main_save_path = main_save_path
        self._main_save_path.mkdir(parents=True, exist_ok=True)
        self._timestamp = timestamp
        self._scenario_type = scenario_type

        if metrics is None:
            self._metrics: List[AbstractMetricBuilder] = []
        else:
            self._metrics = metrics

    @property
    def scenario_type(self) -> str:
        """ Retrieve scenario type. """
        return self._scenario_type

    @property
    def metrics(self) -> List[AbstractMetricBuilder]:
        """ Retrieve a list of metric results. """
        return self._metrics

    def add_metric(self, metric_builder: AbstractMetricBuilder) -> None:
        """todo: need to create the list of types needed from the history"""
        self._metrics.append(metric_builder)

    def save_metric_files(self, metric_files: List[MetricFile]) -> None:
        """
        Save metric results to pickle files.
        :param metric_files: Metric files.
        """

        if not metric_files:
            return

        # Path = main_path/planner/scenario_type/metric_result_name/scenario_name.pkl
        for metric_file in metric_files:
            key = metric_file.key
            save_path = Path.joinpath(self._main_save_path, key.planner_name, key.scenario_type)
            for statistic_name, statistics in metric_file.metric_statistics.items():
                metric_result_path = Path.joinpath(save_path, statistic_name)
                if not metric_result_path.exists():
                    metric_result_path.mkdir(parents=True, exist_ok=True)
                file_name = Path.joinpath(metric_result_path, key.scenario_name + '.pkl')
                with open(file_name, 'wb') as f:
                    pickle.dump(metric_file.serialize(), f)

                logger.debug(f"Saved simulation metric results at: {file_name}.")

    def compute_metric_results(self, history: SimulationHistory) -> Dict[str, List[MetricStatistics]]:
        """
        Compute metrics in the engine.
        :param history: History from simulation.
        :return A list of metric statistics.
        """

        metric_results = {}
        for metric in self._metrics:
            try:
                start_time = time.perf_counter()
                metric_results[metric.name] = metric.compute(history)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.debug(f"Metric: {metric.name} running time: {elapsed_time:.2f} seconds.")
            except (NotImplementedError, Exception) as e:
                # Catch any error when computing a metric.
                logger.error(f"Running {metric.name} with error: {e}")
                raise RuntimeError(f"Metric Engine failed with: {e}")

        return metric_results

    def compute(self, history: SimulationHistory, scenario_name: str, planner_name: str) -> List[MetricFile]:
        """
        Compute metrics and return in a format of MetricStorageResult for each metric computation.
        :param history: History from simulation.
        :param scenario_name: Name of the scenario
        :param planner_name: name of the planner
        :return A list of MetricStorageResult.
        """

        all_metrics_results = self.compute_metric_results(history=history)
        metric_files: List[MetricFile] = []
        for metric_name, results in all_metrics_results.items():
            metric_file_key = MetricFileKey(metric_name=metric_name,
                                            scenario_name=scenario_name,
                                            scenario_type=self.scenario_type,
                                            planner_name=planner_name)
            metric_statistics: Dict[str, List[MetricStatistics]] = defaultdict(list)
            for result in results:
                metric_statistics[result.name].append(result)
            metric_file = MetricFile(key=metric_file_key,
                                     metric_statistics=metric_statistics)
            metric_files.append(metric_file)

        return metric_files
