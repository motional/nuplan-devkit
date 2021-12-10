import logging
import pathlib
from typing import List

import torch
from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.serialization_callback import SerializationCallback
from nuplan.planning.simulation.callback.timing_callback import TimingCallback
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_simulation_callbacks(cfg: DictConfig, output_dir: pathlib.Path) -> List[AbstractCallback]:
    """
    Builds callback.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param output_dir: directory for all experiment results
    :return: List of callbacks.
    """
    logger.info('Building AbstractCallback...')
    callbacks = []
    for callback_name, config in cfg.callback.items():
        if is_target_type(config, SerializationCallback):
            callback: SerializationCallback = instantiate(config, vehicle=cfg.vehicle_parameters,
                                                          output_directory=output_dir)
        elif is_target_type(config, TimingCallback):
            tensorboard = torch.utils.tensorboard.SummaryWriter(log_dir=output_dir)
            callback = instantiate(config, writer=tensorboard)
        else:
            callback = instantiate(config)
        validate_type(callback, AbstractCallback)
        callbacks.append(callback)
    logger.info('Building AbstractCallback...DONE!')
    return callbacks
