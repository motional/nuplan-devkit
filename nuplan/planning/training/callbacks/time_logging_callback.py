import time
from typing import Any, Optional

import pytorch_lightning as pl


class TimeLoggingCallback(pl.Callback):
    """Log training & validation epoch time."""

    def __init__(self) -> None:
        """
        Setup start timestamp.
        """
        self.train_start = 0.0
        self.valid_start = 0.0
        self.test_start = 0.0

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the start of each validation epoch.
        :param trainer: Trainer instance.
        :param pl_module: LightningModule instance.
        """
        self.valid_start = time.time()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of each validation epoch.
        :param trainer: Trainer instance.
        :param pl_module: LightningModule instance.
        """
        pl_module.log_dict({'time_eval': time.time() - self.valid_start, 'step': pl_module.current_epoch})

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the start of each test epoch.
        :param trainer: Trainer instance.
        :param pl_module: LightningModule instance.
        """
        self.test_start = time.time()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of each test epoch.
        :param trainer: Trainer instance.
        :param pl_module: LightningModule instance.
        """
        pl_module.log_dict({'time_test': time.time() - self.test_start, 'step': pl_module.current_epoch})

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the start of each train epoch.
        :param trainer: Trainer instance.
        :param pl_module: LightningModule instance.
        """
        self.train_start = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        """
        Called at the end of each train epoch.
        :param trainer: Trainer instance.
        :param pl_module: LightningModule instance.
        :param outputs: Not required for time logging.
        """
        pl_module.log_dict({'time_epoch': time.time() - self.train_start, 'step': pl_module.current_epoch})
