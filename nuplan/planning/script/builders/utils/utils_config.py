import logging
import math
import os
from pathlib import Path
from shutil import rmtree

import torch
from hydra._internal.instantiate._instantiate2 import _resolve_target
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim.lr_scheduler import OneCycleLR

from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.simulation.callback.timing_callback import TimingCallback

logger = logging.getLogger(__name__)


def update_config_for_training(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: omegaconf dictionary that is used to run the experiment
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)

    if cfg.cache.cache_path is None:
        logger.warning('Parameter cache_path is not set, caching is disabled')
    else:
        if not str(cfg.cache.cache_path).startswith('s3://'):
            if cfg.cache.cleanup_cache and Path(cfg.cache.cache_path).exists():
                rmtree(cfg.cache.cache_path)

            Path(cfg.cache.cache_path).mkdir(parents=True, exist_ok=True)

    if cfg.lightning.trainer.overfitting.enable:
        cfg.data_loader.params.num_workers = 0

    if cfg.gpu and torch.cuda.is_available():
        cfg.lightning.trainer.params.gpus = -1
    else:
        cfg.lightning.trainer.params.gpus = None
        cfg.lightning.trainer.params.accelerator = None
        cfg.lightning.trainer.params.precision = 32

    # Save all interpolations and remove keys that were only used for interpolation and have no further use.
    OmegaConf.resolve(cfg)

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True)

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:
        logger.info(f'Creating experiment name [{cfg.experiment}] in group [{cfg.group}] with config...')
        logger.info('\n' + OmegaConf.to_yaml(cfg))


def update_config_for_simulation(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)
    if cfg.max_number_of_workers:
        # In case simulation is running in multi-threaded way perform the following
        # Remove the locking bottleneck
        cfg.callbacks = [callback for callback in cfg.callback.values() if not is_target_type(callback, TimingCallback)]

    # Save all interpolations and remove keys that were only used for interpolation and have no further use.
    OmegaConf.resolve(cfg)

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True)

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:
        logger.info(f'Creating experiment: {cfg.experiment}')
        logger.info('\n' + OmegaConf.to_yaml(cfg))


def update_config_for_nuboard(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)

    if cfg.simulation_path is None:
        cfg.simulation_path = []
    elif not (isinstance(cfg.simulation_path, list) or isinstance(cfg.simulation_path, ListConfig)):
        cfg.simulation_path = [cfg.simulation_path]

    # Save all interpolations and remove keys that were only used for interpolation and have no further use.
    OmegaConf.resolve(cfg)

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True)

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:
        logger.info('\n' + OmegaConf.to_yaml(cfg))


def update_distributed_optimizer_config(cfg: DictConfig) -> DictConfig:
    """
    Scale the learning rate according to scaling method provided in distributed setting with ddp strategy.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return cfg: DictConfig. Updated configuration that is used to run the experiment.
    """
    if (
        cfg.lightning.distributed_training.lr_scaling_method == 'none'
        or cfg.lightning.trainer.params.accelerator != 'ddp'
    ):
        logger.info('Learning rate will not be scaled.')
        return cfg

    lr_scale = int(os.environ.get('WORLD_SIZE', 1))
    logger.info(f'World size: {lr_scale}')
    logger.info(f'Learning rate before: {cfg.optimizer.lr}')
    logger.info(f'Scaling method: {cfg.lightning.distributed_training.scale_lr}')

    # TODO: support other distributed training strategies
    cfg.optimizer.lr = scale_lr_ddp(
        cfg.optimizer.lr, world_size=lr_scale, scaling_method=cfg.lightning.distributed_training.lr_scaling_method
    )

    logger.info(f'Learning rate after scaling: {cfg.optimizer.lr}')

    return cfg


def scale_lr_ddp(lr: float, world_size: int, scaling_method: str) -> float:
    """
    Scales lr using method provided in the context of PytorchLightning's ddp.
    :param lr: Learning rate provided
    :param world_size: Number gpus used
    :param scaling_method: Method to scale the learning rate by
    :return lr: Learning rate after scaling
    """
    if scaling_method == 'linearly':
        lr *= world_size
    elif scaling_method == 'equal_variance':
        lr *= (world_size) ** 0.5
    else:  # if user specifies none of the currently supported options
        raise (RuntimeError(f'The lr scaling method specified is not supported: {scaling_method}'))
    return lr


def update_distributed_lr_scheduler_config(cfg: DictConfig, train_dataset_len: int) -> DictConfig:
    """
    Adapted from ml_products/lsn/lsn/builders/lr_scheduler.py
    Updates the learning rate scheduler config that modifies optimizer parameters over time.
    Optimizer and LR Scheduler is built in configure_optimizers() methods of the model.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param train_dataset_len: Length of the train dataset.
    :return cfg: Configuration with the updated lr_scheduler key.
    """
    logger.info('Updating Learning Rate Scheduler Config...')

    if cfg.lightning.trainer.params.accelerator != 'ddp':
        return cfg

    number_gpus = int(os.environ.get('WORLD_SIZE', 1))
    # Setup learning rate and momentum schedulers
    if _resolve_target(cfg.lr_scheduler._target_) == OneCycleLR:
        # compute the steps_per_epoch
        cfg.lr_scheduler.steps_per_epoch = scale_oclr_steps_per_epoch_ddp(
            batch_size=cfg.data_loader.params.batch_size,
            overfit_batches=cfg.lightning.trainer.overfitting.params.overfit_batches,
            train_dataset_len=train_dataset_len,
            world_size=number_gpus,
        )

        # Ensure the initial learning rate used is correct by adjusting max_lr
        # This only has to be done for OneCycleLR which overrides the lr in the optimizer
        # provided with max_lr/div_factor
        div_factor = cfg.lr_scheduler.div_factor  # factor to divide the max_lr by to get the initial lr
        cfg.lr_scheduler.max_lr = cfg.optimizer.lr * div_factor

        logger.info('Updating Learning Rate Scheduler Config Completed. Using {cfg.lr_scheduler._target_}')

    return cfg  # return cfg is there was no optimizer override


def scale_oclr_steps_per_epoch_ddp(
    batch_size: int, overfit_batches: float, train_dataset_len: int, world_size: int
) -> int:
    """
    Scales lr using method provided in the context of PytorchLightning's ddp.
    :param batch_size: Batch size that each GPU sees in ddp
    :param overfit_batches: Number of batches to overfit. Could be integer or fraction
    :param world_size: Number gpus used
    :return steps_per_epoch_per_gpu: Step per epoch after scaling
    """
    if overfit_batches == 0.0:
        num_samples_per_gpu = math.ceil(train_dataset_len / world_size)
        steps_per_epoch_per_gpu = math.ceil(num_samples_per_gpu / batch_size)
    elif overfit_batches >= 1.0:
        steps_per_epoch_per_gpu = math.ceil(overfit_batches / world_size)
    else:  # fractional overfit_batches
        overfit_train_dataset_len = math.ceil(train_dataset_len * overfit_batches)
        num_samples_per_gpu = math.ceil(overfit_train_dataset_len / world_size)
        steps_per_epoch_per_gpu = math.ceil(num_samples_per_gpu / batch_size)
    return steps_per_epoch_per_gpu
