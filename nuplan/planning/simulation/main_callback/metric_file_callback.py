import logging
import pathlib
import time
from collections import defaultdict
from typing import List

import pandas

from nuplan.common.utils.io_utils import (
    delete_file,
    list_files_in_directory,
    path_exists,
    read_pickle,
    safe_path_to_string,
)
from nuplan.common.utils.s3_utils import is_s3_path
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
        Output path can be local or s3.
        :param metric_file_output_path: Path to save integrated metric files.
        :param scenario_metric_paths: A list of paths with scenario metric files.
        :param delete_scenario_metric_files: Set True to delete scenario metric files.
        """
        self._metric_file_output_path = pathlib.Path(metric_file_output_path)
        if not is_s3_path(self._metric_file_output_path):
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
            # If it's an dir in S3, path_exists will be False but there may be files
            if not is_s3_path(scenario_metric_path) and not path_exists(scenario_metric_path):
                continue

            for scenario_metric_file in list_files_in_directory(scenario_metric_path):
                if not scenario_metric_file.name.endswith(JSON_FILE_EXTENSION):
                    continue

                json_dataframe = read_pickle(scenario_metric_file)
                for dataframe in json_dataframe:
                    pandas_dataframe = pandas.DataFrame(dataframe)
                    metrics[dataframe['metric_statistics_name']].append(pandas_dataframe)

                # Delete the temp file
                if self._delete_scenario_metric_files:
                    delete_file(scenario_metric_file)

        for metric_statistics_name, dataframe in metrics.items():
            save_path = self._metric_file_output_path / (metric_statistics_name + '.parquet')
            concat_pandas = pandas.concat([*dataframe], ignore_index=True)
            concat_pandas.to_parquet(safe_path_to_string(save_path))

        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))
        logger.info(f"Metric files integration: {time_str} [HH:MM:SS]")
