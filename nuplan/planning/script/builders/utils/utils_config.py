import logging
from pathlib import Path
from shutil import rmtree

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

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
        cfg.callbacks = [callback for callback in cfg.callbacks if not is_target_type(callback, TimingCallback)]
        # Make sure we dump scenarios only at the end of simulation in order to avoid I/O bottleneck
        cfg.callback.serialization_callback.serialize_after_every_simulation_step = False

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
