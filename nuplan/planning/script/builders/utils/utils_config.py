import logging
import math
import os
from pathlib import Path
from shutil import rmtree
from typing import cast

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim.lr_scheduler import OneCycleLR

from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.simulation.callback.timing_callback import TimingCallback
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool

logger = logging.getLogger(__name__)


def update_config_for_training(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: omegaconf dictionary that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)

    if cfg.cache.cache_path is None:
        logger.warning("Parameter cache_path is not set, caching is disabled")
    else:
        if not str(cfg.cache.cache_path).startswith("s3://"):
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
        logger.info(f"Creating experiment name [{cfg.experiment}] in group [{cfg.group}] with config...")
        logger.info("\n" + OmegaConf.to_yaml(cfg))


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
        logger.info(f"Creating experiment: {cfg.experiment}")
        logger.info("\n" + OmegaConf.to_yaml(cfg))


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
        logger.info("\n" + OmegaConf.to_yaml(cfg))


def update_distributed_optimizer_config(cfg: DictConfig) -> DictConfig:
    """
    Scale the learning rate according to scaling method provided in distributed setting with ddp strategy.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return cfg: DictConfig. Updated configuration that is used to run the experiment.
    """
    lr_scale = get_num_gpus_used(cfg)
    logger.info(f"World size: {lr_scale}")
    logger.info(f"Learning rate before: {cfg.optimizer.lr}")
    scaling_method = (
        "Equal Variance" if cfg.lightning.distributed_training.equal_variance_scaling_strategy else "Linearly"
    )
    logger.info(f"Scaling method: {scaling_method}")

    # TODO: support other distributed training strategies like ddp2, dp, etc
    cfg.optimizer.lr = scale_parameter(
        parameter=cfg.optimizer.lr,
        world_size=lr_scale,
        equal_variance_scaling_strategy=cfg.lightning.distributed_training.equal_variance_scaling_strategy,
    )
    # if optimizer is Adam or AdamW, scale the momentum as well
    if is_target_type(cfg.optimizer, torch.optim.Adam) or is_target_type(cfg.optimizer, torch.optim.AdamW):
        cfg.optimizer.betas[0] = scale_parameter(
            parameter=cfg.optimizer.betas[0],
            world_size=lr_scale,
            equal_variance_scaling_strategy=cfg.lightning.distributed_training.equal_variance_scaling_strategy,
            raise_power=True,
        )
        cfg.optimizer.betas[1] = scale_parameter(
            parameter=cfg.optimizer.betas[1],
            world_size=lr_scale,
            equal_variance_scaling_strategy=cfg.lightning.distributed_training.equal_variance_scaling_strategy,
            raise_power=True,
        )
        logger.info(f"Betas after scaling: {cfg.optimizer.betas}")

    logger.info(f"Learning rate after scaling: {cfg.optimizer.lr}")

    return cfg


def scale_parameter(
    parameter: float, world_size: int, equal_variance_scaling_strategy: bool, raise_power: bool = False
) -> float:
    """
    Scale parameter (such as learning rate or beta values in Adam/AdamW optimizer) using method specified in the context of PytorchLightning's ddp.
    :param parameter: Learning rate/beta values used in Adam optimizer/etc.
    :param world_size: Number gpus used.
    :param equal_variance_scaling_strategy: Whether the method to scale the learning rate or betas by is equal_variance (by square root of num GPUs); otherwise it is linearly (by num GPUs).
    :return parameter: Learning rate/beta values used in Adam optimizer/etc after scaling.
    """
    scaling_factor = world_size**0.5 if equal_variance_scaling_strategy else world_size
    parameter = parameter * scaling_factor if not raise_power else parameter**scaling_factor

    return parameter


def update_distributed_lr_scheduler_config(cfg: DictConfig, num_train_batches: int) -> DictConfig:
    """
    Updates the learning rate scheduler config that modifies optimizer parameters over time.
    Optimizer and LR Scheduler is built in configure_optimizers() methods of the model.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param num_train_batches: Number of batches in train dataloader.
    :return cfg: Configuration with the updated lr_scheduler key.
    """
    logger.info("Updating Learning Rate Scheduler Config...")

    number_gpus = get_num_gpus_used(cfg)
    # Setup learning rate and momentum schedulers

    if is_target_type(cfg.lr_scheduler, OneCycleLR):
        enable_overfitting = cfg.lightning.trainer.overfitting.enable
        overfit_batches = cfg.lightning.trainer.overfitting.params.overfit_batches

        if enable_overfitting and overfit_batches != 0:  # if overfitting, number of batches used in training changes

            if overfit_batches >= 1.0:  # if number of batches to overfit is integer value
                num_train_batches = overfit_batches

            else:  # if fraction is given as input to overfit_batches
                num_train_batches = math.ceil(num_train_batches * overfit_batches)

        cfg.lr_scheduler.steps_per_epoch = scale_oclr_steps_per_epoch(
            num_train_batches=num_train_batches,
            world_size=number_gpus,
            epochs=cfg.lightning.trainer.params.max_epochs,
            warm_up_steps=cfg.warm_up_lr_scheduler.lr_lambda.warm_up_steps if "warm_up_lr_scheduler" in cfg else 0,
        )

        logger.info(f"Updating learning rate scheduler config Completed. Using {cfg.lr_scheduler._target_}.")

    else:  # Only updating of OneCycleLR is supported as of right now
        logger.info(
            f"Updating {cfg.lr_scheduler._target_} in ddp setting is not yet supported. Learning rate scheduler config will not be updated."
        )

    return cfg


def scale_oclr_steps_per_epoch(num_train_batches: int, world_size: int, epochs: int, warm_up_steps: int) -> int:
    """
    Scales OneCycleLR steps per epoch using method provided in the context of PytorchLightning's ddp.
    :param num_train_batches: Number of batches in train_dataloader.
    :param world_size: Number gpus used.
    :param epochs: Number of epochs we are training for.
    :param warm_up_steps: Number of warm up steps in the warm up scheduler used before OneCycleLR si used.
    :return steps_per_epoch_per_gpu_after_warm_up: Step per epoch after scaling and taking into account warm up steps.
    """
    num_batches_per_gpu = math.ceil(num_train_batches / world_size)
    steps_per_epoch_per_gpu = math.ceil(num_batches_per_gpu / epochs)
    total_steps_per_gpu = steps_per_epoch_per_gpu * epochs - warm_up_steps  # take into account warm_up_steps
    steps_per_epoch_per_gpu_after_warm_up = math.ceil(total_steps_per_gpu / epochs)

    return steps_per_epoch_per_gpu_after_warm_up


def scale_cfg_for_distributed_training(
    cfg: DictConfig, datamodule: pl.LightningDataModule, worker: WorkerPool
) -> DictConfig:
    """
    Adjusts parameters in cfg for ddp.
    :param cfg: Config with parameters for instantiation.
    :param datamodule: Datamodule which will be used for updating the lr_scheduler parameters.
    :return cfg: Updated config.
    """
    OmegaConf.set_struct(cfg, False)
    cfg = update_distributed_optimizer_config(cfg)
    # Update lr_scheduler with yaml file config before building lightning module
    if "lr_scheduler" in cfg:
        num_train_samples = int(
            len(datamodule._splitter.get_train_samples(datamodule._all_samples, worker)) * datamodule._train_fraction
        )
        cfg = update_distributed_lr_scheduler_config(
            cfg=cfg,
            num_train_batches=num_train_samples // cfg.data_loader.params.batch_size,
        )

    OmegaConf.set_struct(cfg, True)
    logger.info("Optimizer and LR Scheduler configs updated according to ddp strategy.")
    return cfg


def get_num_gpus_used(cfg: DictConfig) -> int:
    """
    Gets the number of gpus used in ddp by searching through the environment variable WORLD_SIZE, PytorchLightning Trainer specified number of GPUs, and torch.cuda.device_count() in that order.
    :param cfg: Config with experiment parameters.
    :return num_gpus: Number of gpus used in ddp.
    """
    num_gpus = os.getenv("WORLD_SIZE", -1)

    if num_gpus == -1:  # if environment variable WORLD_SIZE is not set, find from trainer
        logger.info("WORLD_SIZE was not set.")
        trainer_num_gpus = cfg.lightning.trainer.params.gpus

        if isinstance(trainer_num_gpus, str):
            raise RuntimeError("Error, please specify gpus as integer. Received string.")
        trainer_num_gpus = cast(int, trainer_num_gpus)

        if trainer_num_gpus == -1:  # if trainer gpus = -1, all gpus are used, so find all available devices
            logger.info(
                "PytorchLightning Trainer gpus was set to -1, finding number of GPUs used from torch.cuda.device_count()."
            )
            cuda_num_gpus = torch.cuda.device_count() * int(os.getenv("NUM_NODES", 1))
            num_gpus = cuda_num_gpus

        else:  # if trainer gpus is not -1
            logger.info(f"Trainer gpus was set to {trainer_num_gpus}, using this as the number of gpus.")
            num_gpus = trainer_num_gpus

    num_gpus = int(num_gpus)
    logger.info(f"Number of gpus found to be in use: {num_gpus}")
    return num_gpus
