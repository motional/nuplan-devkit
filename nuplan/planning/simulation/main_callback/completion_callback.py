import logging
import os
from pathlib import Path

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

logger = logging.getLogger(__name__)


class CompletionCallback(AbstractMainCallback):
    """Callback that creates a token file to mark that the simulation instance finished the job."""

    def __init__(self, output_dir: str, challenge_name: str):
        """
        :param output_dir: Root dir used to find the report file and as path to save results.
        :param challenge_name: Name of the challenge being run.
        """
        self._bucket = os.getenv("NUPLAN_SERVER_S3_ROOT_URL")
        assert self._bucket, "Target bucket must be specified!"

        instance_id = os.getenv("SCENARIO_FILTER_ID", "0")

        task_id = '_'.join([challenge_name, instance_id])
        self._completion_dir = Path(output_dir, 'simulation-results', task_id)

    def on_run_simulation_end(self) -> None:
        """
        On reached_end mark the task as completed by creating the relative file.
        """
        self._write_empty_file(self._completion_dir, 'completed.txt')

    @staticmethod
    def _write_empty_file(path: Path, filename: str) -> None:
        """
        Creates an empty file with the specified name at the given location.
        :param path: The location where to create the file.
        :param filename: The name of the file to be created.
        """
        if not is_s3_path(path):
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Writing file {path/filename}')
        with (path / filename).open('w'):
            # Writes an empty file.
            pass
