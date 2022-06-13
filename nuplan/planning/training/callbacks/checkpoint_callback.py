from pathlib import Path
from typing import Optional

import pytorch_lightning as pl


class ModelCheckpointAtEpochEnd(pl.callbacks.ModelCheckpoint):
    """Customized callback for saving Lightning checkpoint for every epoch."""

    def __init__(
        self,
        save_top_k: int = -1,
        save_last: bool = False,
        dirpath: Optional[str] = None,
        monitor: Optional[str] = None,
        mode: str = 'max',
    ):
        """
        Initialize the callback
        :param save_top_k: Choose how many best checkpoints we want to save:
            save_top_k == 0 means no models are saved.
            save_top_k == -1 means all models are saved.
        :param save_last: Whether to save the last model as last.ckpt.
        :param dirpath: Directory where the checkpoints are saved.
        :param monitor: The metrics to monitor for saving best checkpoints.
        :param mode: How we want to choose the best model: min, max or auto for the metrics we choose.
        """
        super().__init__(save_last=save_last, save_top_k=save_top_k, dirpath=dirpath, monitor=monitor, mode=mode)

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Customized callback function to save checkpoint every epoch.
        :param trainer: Pytorch lightning trainer instance.
        :param pl_module: LightningModule.
        """
        checkpoint_dir = Path(trainer.checkpoint_callback.dirpath).parent / 'checkpoints'
        checkpoint_name = f'epoch={trainer.current_epoch}.ckpt'
        checkpoint_path = checkpoint_dir / checkpoint_name
        trainer.save_checkpoint(str(checkpoint_path))


class EvaluationResumeCallback(pl.Callback):
    """Resumes evaluation at the specified epoch number."""

    def __init__(self, epoch_to_resume: int):
        """
        Initialize the callback.
        :param epoch_to_resume: The epoch count of previous evaluation.
        """
        self.epoch_to_resume = epoch_to_resume
        assert self.epoch_to_resume >= 0, f"Invalid epoch number to resume: {self.epoch_to_resume}"
        self._run_eval = True

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when starting validation.
        :param trainer: The current pytorch_lightning.trainer.Trainer instance.
        :param pl_module: The current pytorch_lightning.core.lightning.LightningModule instance.
        """
        # Inject evaluation epoch to trainer and start evaluation logging
        if self._run_eval:
            if trainer.current_epoch == 0:
                # Restore training states from the checkpoint.
                # trainer.validate() doesn't load the checkpoint when a model is provided.
                trainer.checkpoint_connector.restore_weights()
            trainer.current_epoch = self.epoch_to_resume

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when finishing validation.
        :param trainer: the current pytorch_lightning.trainer.Trainer instance.
        :param pl_module: the current pytorch_lightning.core.lightning.LightningModule instance.
        """
        # Turn off epoch resuming.
        if self._run_eval:
            self._run_eval = False

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when starting testing.
        :param trainer: The current pytorch_lightning.trainer.Trainer instance.
        :param pl_module: The current pytorch_lightning.core.lightning.LightningModule instance.
        """
        self.on_validation_start(trainer, pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when finishing testing.
        :param trainer: The current pytorch_lightning.trainer.Trainer instance.
        :param pl_module: The current pytorch_lightning.core.lightning.LightningModule instance.
        """
        self.on_validation_end(trainer, pl_module)
