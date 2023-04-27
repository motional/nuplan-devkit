import logging
import time
from pathlib import Path
from typing import List

from nuplan.common.utils.io_utils import list_files_in_directory
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.metrics.aggregator.abstract_metric_aggregator import AbstractMetricAggregator
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

logger = logging.getLogger(__name__)


class MetricAggregatorCallback(AbstractMainCallback):
    """Callback to aggregate metrics after the simulation ends."""

    def __init__(self, metric_save_path: str, metric_aggregators: List[AbstractMetricAggregator]):
        """Callback to handle metric files at the end of process."""
        self._metric_save_path = Path(metric_save_path)
        self._metric_aggregators = metric_aggregators

    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        start_time = time.perf_counter()

        # Stop if no metric save path
        if not is_s3_path(self._metric_save_path) and not self._metric_save_path.exists():
            return

        # Run a list of metric aggregators
        for metric_aggregator in self._metric_aggregators:
            # Load metric parquet files
            metric_dataframes = {}

            if is_s3_path(self._metric_save_path):
                metrics = [
                    path for path in list_files_in_directory(self._metric_save_path) if path.suffix == ".parquet"
                ]
            else:
                metrics = list(self._metric_save_path.rglob("*.parquet"))

            if not metric_aggregator.challenge:
                challenge_metrics = list(metrics)
            else:
                challenge_metrics = [path for path in metrics if metric_aggregator.challenge in str(path)]

            for file in challenge_metrics:
                # Skip loading if the file cannot be loaded as parquet format
                try:
                    metric_statistic_dataframe = MetricStatisticsDataFrame.load_parquet(file)
                    metric_statistic_name = metric_statistic_dataframe.metric_statistic_name
                    metric_dataframes[metric_statistic_name] = metric_statistic_dataframe
                except (FileNotFoundError, Exception) as e:
                    logger.info(f"Cannot load the file: {file}, error: {e}")

            if metric_dataframes:
                logger.info(f"Running metric aggregator: {metric_aggregator.name}")
                metric_aggregator(metric_dataframes=metric_dataframes)
            else:
                logger.warning(f"{metric_aggregator.name}: No metric files found for aggregation!")
                logger.warning(
                    "If you didn't expect this, ensure that the challenge name is part of your submitted job name."
                )

        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))
        logger.info(f"Metric aggregator: {time_str} [HH:MM:SS]")
