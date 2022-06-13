import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor

logger = logging.getLogger(__name__)


def build_agent_augmentor(cfg: DictConfig) -> List[AbstractAugmentor]:
    """
    Build list of augmentors based on config.
    :param cfg: Dict config.
    :return List of augmentor objects.
    """
    logger.info('Building augmentors...')

    instantiated_augmentors = []

    for augmentor_type in cfg.values():
        augmentor: AbstractAugmentor = instantiate(augmentor_type)
        validate_type(augmentor, AbstractAugmentor)
        instantiated_augmentors.append(augmentor)

    logger.info('Building augmentors...DONE!')

    return instantiated_augmentors
