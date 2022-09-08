import logging
import pathlib
import pickle
import time
from collections import defaultdict
from typing import List

import pandas

from nuplan.planning.metrics.metric_engine import JSON_FILE_EXTENSION
from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

logger = logging.getLogger(__name__)


class MetricFileCallback(AbstractMainCallback):
    """Callback to handle metric files at the end of process."""

    def __init__(
        self, metric_file_output_path: str, scenario_metric_paths: List[str], delete_scenario_metric_files: bool = False
    ):
        """
        Constructor of MetricFileCallback.
        :param metric_file_output_path: Path to save integrated metric files.
        :param scenario_metric_paths: A list of paths with scenario metric files.
        :param delete_scenario_metric_files: Set True to delete scenario metric files.
        """
        self._metric_file_output_path = pathlib.Path(metric_file_output_path)
        if not self._metric_file_output_path.exists():
            self._metric_file_output_path.mkdir(exist_ok=True, parents=True)

        self._scenario_metric_paths = [
            pathlib.Path(scenario_metric_path) for scenario_metric_path in scenario_metric_paths
        ]
        self._delete_scenario_metric_files = delete_scenario_metric_files

    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        start_time = time.perf_counter()

        # Integrate scenario metric files into metric statistic files
        metrics = defaultdict(list)

        # Stop if no metric path exists
        for scenario_metric_path in self._scenario_metric_paths:
            if not scenario_metric_path.exists():
                continue

            for scenario_metric_file in scenario_metric_path.iterdir():
                if not scenario_metric_file.name.endswith(JSON_FILE_EXTENSION):
                    continue
                with open(scenario_metric_file, "rb") as f:
                    json_dataframe = pickle.load(f)
                    for dataframe in json_dataframe:
                        pandas_dataframe = pandas.DataFrame(dataframe)
                        metrics[dataframe['metric_statistics_name']].append(pandas_dataframe)

                # Delete the temp file
                if self._delete_scenario_metric_files:
                    scenario_metric_file.unlink(missing_ok=True)

        for metric_statistics_name, dataframe in metrics.items():
            save_path = self._metric_file_output_path / (metric_statistics_name + '.parquet')
            concat_pandas = pandas.concat([*dataframe], ignore_index=True)
            concat_pandas.to_parquet(save_path)

        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))
        logger.info(f"Metric files integration: {time_str} [HH:MM:SS]")
