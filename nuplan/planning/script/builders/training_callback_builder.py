import logging
from typing import List

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """
    Build callbacks based on config.
    :param cfg: Dict config.
    :return List of callbacks.
    """
    logger.info('Building callbacks...')

    instantiated_callbacks = []

    for callback_type in cfg.values():
        callback: pl.Callback = instantiate(callback_type)
        validate_type(callback, pl.Callback)
        instantiated_callbacks.append(callback)

    logger.info('Building callbacks...DONE!')

    return instantiated_callbacks
