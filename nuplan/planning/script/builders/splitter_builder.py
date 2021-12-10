import logging

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_splitter(cfg: DictConfig) -> AbstractSplitter:
    """
    Builds the splitter.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param db: dabatase
    :return: Instance of Splitter.
    """

    logger.info('Building Splitter...')
    splitter: AbstractSplitter = instantiate(cfg)
    validate_type(splitter, AbstractSplitter)
    logger.info('Building Splitter...DONE!')
    return splitter
