import logging
import pathlib

import pytorch_lightning as pl
from pyinstrument import Profiler

from nuplan.common.utils.s3_utils import is_s3_path

logger = logging.getLogger(__name__)


class ProfileCallback(pl.Callback):
    """Profiling callback that produces an html report."""

    def __init__(self, output_dir: pathlib.Path, interval: float = 0.01):
        """
        Initialize callback.
        :param output_dir: directory where output should be stored. Note, "profiling" sub-dir will be added
        :param interval: of the profiler
        """
        # Set directory
        self._output_dir = output_dir / "profiling"
        if not is_s3_path(self._output_dir):
            self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Profiler will report into folder: {str(self._output_dir)}")

        # Create Profiler
        self._profiler = Profiler(interval=interval)
        self._profiler_running = False

    def on_init_start(self, trainer: pl.Trainer) -> None:
        """
        Called during training initialization.
        :param trainer: Lightning trainer.
        """
        self.start_profiler("on_init_start")

    def on_init_end(self, trainer: pl.Trainer) -> None:
        """
        Called at the end of the training.
        :param trainer: Lightning trainer.
        """
        self.save_profiler("on_init_end")

    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at each epoch start.
        :param trainer: Lightning trainer.
        :param pl_module: lightning model.
        """
        self.start_profiler("on_epoch_start")

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at each epoch end.
        :param trainer: Lightning trainer.
        :param pl_module: lightning model.
        """
        self.save_profiler("epoch_" + str(trainer.current_epoch) + "-on_epoch_end")

    def start_profiler(self, when: str) -> None:
        """
        Start the profiler.
        Raise: in case profiler is already running.
        :param when: Message to log when starting the profiler.
        """
        assert not self._profiler_running, "Profiler can not be started twice!"
        logger.info(f"STARTING profiler: {when}")
        self._profiler_running = True
        self._profiler.start()

    def stop_profiler(self) -> None:
        """
        Start profiler
        Raise: in case profiler is not running
        """
        assert self._profiler_running, "Profiler has to be running!!"
        self._profiler.stop()
        self._profiler_running = False

    def save_profiler(self, file_name: str) -> None:
        """
        Save profiling output to a html report
        :param file_name: File name to save report to.
        """
        self.stop_profiler()

        # Save output
        profiler_out_html = self._profiler.output_html()

        # Create path
        html_save_path = self._output_dir / file_name
        path = str(html_save_path.with_suffix(".html"))
        logger.info(f"Saving profiler output to: {path}")

        fp = open(path, "w+")
        fp.write(profiler_out_html)
        fp.close()
