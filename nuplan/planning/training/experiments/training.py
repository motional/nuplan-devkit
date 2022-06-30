import logging
import pathlib
from dataclasses import dataclass

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.training_builder import (
    build_lightning_datamodule,
    build_lightning_module,
    build_trainer,
)
from nuplan.planning.script.builders.utils.utils_config import (
    update_distributed_lr_scheduler_config,
    update_distributed_optimizer_config,
)
from nuplan.planning.training.callbacks.profile_callback import ProfileCallback
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingEngine:
    """Lightning training engine dataclass wrapping the lightning trainer, model and datamodule."""

    trainer: pl.Trainer  # Trainer for models
    model: pl.LightningModule  # Module describing NN model, loss, metrics, visualization
    datamodule: pl.LightningDataModule  # Loading data

    def __repr__(self) -> str:
        """
        :return: String representation of class without expanding the fields.
        """
        return f"<{type(self).__module__}.{type(self).__qualname__} object at {hex(id(self))}>"


def build_training_engine(cfg: DictConfig, worker: WorkerPool) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: TrainingEngine
    """
    logger.info('Building training engine...')

    # Construct profiler
    profiler = ProfileCallback(pathlib.Path(cfg.output_dir)) if cfg.enable_profiling else None

    # Start profiler if enabled
    if profiler:
        profiler.start_profiler("build_training_engine")

    # Create model
    torch_module_wrapper = build_torch_module_wrapper(cfg.model)

    # Build the datamodule
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)

    # Update the learning rate parameters in multinode setting
    OmegaConf.set_struct(cfg, False)
    cfg = update_distributed_optimizer_config(cfg)
    OmegaConf.set_struct(cfg, True)

    # Update lr_scheduler with yaml file config before building lightning module
    if 'lr_scheduler' in cfg:
        datamodule.setup('fit')  # set up datamodule in order to get length of train_set for updating cfg
        OmegaConf.set_struct(cfg, False)
        cfg = update_distributed_lr_scheduler_config(
            cfg=cfg,
            train_dataset_len=len(datamodule.train_dataloader()),
        )
        OmegaConf.set_struct(cfg, True)

    # Build lightning module
    model = build_lightning_module(cfg, torch_module_wrapper)

    # Build trainer
    trainer = build_trainer(cfg)

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    # Save profiler output
    if profiler:
        profiler.save_profiler("build_training_engine")

    return engine
