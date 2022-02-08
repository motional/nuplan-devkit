import logging
from pathlib import Path
from typing import cast

import pytorch_lightning as pl
import pytorch_lightning.loggers
import pytorch_lightning.plugins
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.script.builders.objectives_builder import build_objectives
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.splitter_builder import build_splitter
from nuplan.planning.script.builders.training_metrics_builder import build_training_metrics
from nuplan.planning.script.builders.utils.utils_checkpoint import extract_last_checkpoint_from_experiment
from nuplan.planning.training.callbacks.checkpoint_callback import ModelCheckpointAtEpochEnd
from nuplan.planning.training.callbacks.time_logging_callback import TimeLoggingCallback
from nuplan.planning.training.data_loader.datamodule import DataModule
from nuplan.planning.training.modeling.nn_model import NNModule
from nuplan.planning.training.modeling.planning_module import PlanningModule
from nuplan.planning.training.preprocessing.feature_caching_preprocessor import FeatureCachingPreprocessor
from nuplan.planning.training.visualization.visualization_callbacks import RasterVisualizationCallback
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def build_lightning_datamodule(cfg: DictConfig, scenario_builder: AbstractScenarioBuilder,
                               worker: WorkerPool, model: NNModule) -> pl.LightningDataModule:
    """
    Builds the lightning datamodule from the config.

    :param cfg: omegaconf dictionary
    :param scenario_builder: access to database able to filter scenarios
    :param model: NN model used for training
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: built object.
    """

    # Build features and targets
    feature_builders = model.get_list_of_required_feature()
    target_builder = model.get_list_of_computed_target()

    # Build splitter
    splitter = build_splitter(cfg.splitter)

    # Create caching feature computator
    computator = FeatureCachingPreprocessor(
        cache_dir=cfg.cache_dir,
        force_feature_computation=cfg.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builder,
    )

    # Build and run scenario filter to extract scenarios
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    logger.info("Extracting all scenarios...")
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
    logger.info("Extracting all scenarios...DONE")

    assert len(scenarios) > 0, "No scenarios were retrieved for training, check the scenario_filter parameters!"

    # Create datamodule
    datamodule: pl.LightningDataModule = DataModule(
        feature_and_targets_builders=computator,
        splitter=splitter,
        all_scenarios=scenarios,
        dataloader_params=cfg.data_loader.params,
        **cfg.data_loader.datamodule,
    )

    return datamodule


def build_lightning_module(cfg: DictConfig, nn_model: NNModule) -> pl.LightningModule:
    """
    Builds the lightning module from the config.

    :param cfg: omegaconf dictionary
    :param nn_model: NN model used for training
    :return: built object.
    """

    # Build loss
    objectives = build_objectives(cfg)

    # Build metrics to evaluate the performance of predictions
    metrics = build_training_metrics(cfg)

    # Create the complete Module
    model = PlanningModule(
        model=nn_model,
        objectives=objectives,
        metrics=metrics,
        **cfg.lightning.hparams,
    )

    return cast(pl.LightningModule, model)


def build_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Builds the lightning trainer from the config.

    :param cfg: omegaconf dictionary
    :return: built object.
    """
    params = cfg.lightning.trainer.params

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True),
        RasterVisualizationCallback(**cfg.lightning.callbacks.raster_visualization),
        TimeLoggingCallback(),
        ModelCheckpointAtEpochEnd(
            dirpath=str(Path(cfg.output_dir) / 'best_model'),
            save_last=False,
            **cfg.lightning.trainer.checkpoint,
        ),
    ]

    if params.gpus:
        callbacks.append(pl.callbacks.GPUStatsMonitor(intra_step_time=True, inter_step_time=True))

    plugins = [
        pl.plugins.DDPPlugin(find_unused_parameters=False),
    ]

    loggers = [
        pl.loggers.TensorBoardLogger(
            save_dir=cfg.group,
            name=cfg.experiment,
            log_graph=False,
            version='',
            prefix='',
        ),
    ]

    if cfg.lightning.trainer.overfitting.enable:
        OmegaConf.set_struct(cfg, False)
        params = OmegaConf.merge(params, cfg.lightning.trainer.overfitting.params)
        params.check_val_every_n_epoch = params.max_epochs + 1
        OmegaConf.set_struct(cfg, True)

        return pl.Trainer(plugins=plugins, **params)

    if cfg.resume_training:
        output_dir = Path(cfg.output_dir)
        date_format = cfg.date_format

        OmegaConf.set_struct(cfg, False)
        last_checkpoint = extract_last_checkpoint_from_experiment(output_dir, date_format)
        if not last_checkpoint:
            raise ValueError("Resume Training is enabled but no checkpoint was found!")

        params.resume_from_checkpoint = str(last_checkpoint)
        logger.info(f'Resuming from checkpoint {last_checkpoint}')

        OmegaConf.set_struct(cfg, True)

    trainer = pl.Trainer(
        callbacks=callbacks,
        plugins=plugins,
        logger=loggers,
        **params,
    )

    return trainer
