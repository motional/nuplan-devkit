import logging
import pathlib
from dataclasses import dataclass

import pytorch_lightning as pl
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.script.builders.model_builder import build_nn_model
from nuplan.planning.script.builders.training_builder import build_lightning_datamodule, build_lightning_module, \
    build_trainer
from nuplan.planning.training.callbacks.profile_callback import ProfileCallback
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingEngine:
    trainer: pl.Trainer  # Trainer for models
    model: pl.LightningModule  # Module describing NN model, loss, metrics, visualization
    datamodule: pl.LightningDataModule  # Loading data


def build_training_engine(cfg: DictConfig,
                          worker: WorkerPool,
                          scenario_builder: AbstractScenarioBuilder) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer

    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :param scenario_builder: access to database able to filter scenarios
    :return: TrainingEngine
    """
    logger.info('Building training engine...')

    # Construct profiler if desired
    profiler = ProfileCallback(pathlib.Path(cfg.output_dir)) if cfg.enable_profiling else None

    # Profile if desired
    if profiler:
        profiler.start_profiler("build_training_engine")

    # Create the model that will be trained
    nn_model = build_nn_model(cfg.model)

    # Build the datamodule
    datamodule = build_lightning_datamodule(cfg, scenario_builder, worker, nn_model)

    # Build LightningModule
    model = build_lightning_module(cfg, nn_model)

    # Build Trainer
    trainer = build_trainer(cfg)

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    # Save profiler output
    if profiler:
        profiler.save_profiler("build_training_engine")

    return engine


def cache_data(cfg: DictConfig, worker: WorkerPool, scenario_builder: AbstractScenarioBuilder) -> None:
    """
    Builds the lightning datamodule and caches all samples.

    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :param scenario_builder: access to database able to filter scenarios
    """
    OmegaConf.set_struct(cfg, False)
    cfg.data_loader.params.batch_size = 1
    cfg.data_loader.params.pin_memory = False
    OmegaConf.set_struct(cfg, True)

    planning_module = build_nn_model(cfg.model)
    datamodule = build_lightning_datamodule(
        cfg=cfg, scenario_builder=scenario_builder, worker=worker, model=planning_module)
    datamodule.setup('fit')
    datamodule.setup('test')

    logger.info('Starting caching...')

    for _ in tqdm(datamodule.train_dataloader()):
        pass

    for _ in tqdm(datamodule.val_dataloader()):
        pass

    for _ in tqdm(datamodule.test_dataloader()):
        pass
