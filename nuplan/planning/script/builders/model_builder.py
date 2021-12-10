import logging

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.modeling.nn_model import NNModule
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_nn_model(cfg: DictConfig) -> NNModule:
    """
    Builds the NN module.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of NNModule.
    """

    logger.info('Building NNModule...')
    model = instantiate(cfg)
    validate_type(model, NNModule)
    logger.info('Building NNModule...DONE!')

    return model
