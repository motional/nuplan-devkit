import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

logger = logging.getLogger(__name__)


def _validation_succeeded(source_folder_path: Path) -> bool:
    """
    Reads runners report and checks if the simulation was successful or not.
    :param source_folder_path:  Root folder to where runners report is stored.
    :return: True, if the simulation was successful, false otherwise.
    """
    try:
        df = pd.read_parquet(f'{source_folder_path}/runner_report.parquet')
    except FileNotFoundError:
        logger.warning("No runners report file found in %s!" % source_folder_path)
        return False

    return bool(np.all(df['succeeded'].values))


class ValidationCallback(AbstractMainCallback):
    """Callback checking if a validation simulation was successful or not."""

    def __init__(self, output_dir: str, validation_dir_name: str):
        """
        :param output_dir: Root dir used to find the report file and as path to save results.
        :param validation_dir_name: Name of the directory where the validation file should be stored.
        """
        self.output_dir = Path(output_dir)
        self._validation_dir_name = validation_dir_name

    def on_run_simulation_end(self) -> None:
        """
        On reached_end push results to S3 bucket.
        """
        if _validation_succeeded(self.output_dir):
            filename = 'passed.txt'
        else:
            filename = 'failed.txt'
        logger.info("Validation filename: %s" % filename)
        validation_dir = self.output_dir / self._validation_dir_name
        if not is_s3_path(validation_dir):
            validation_dir.mkdir(parents=True, exist_ok=True)

        with (validation_dir / filename).open('w'):
            pass
