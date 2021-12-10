import logging

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    Builds the worker.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of WorkerPool.
    """

    logger.info('Building WorkerPool...')
    worker: WorkerPool = instantiate(cfg.worker)
    validate_type(worker, WorkerPool)
    logger.info('Building WorkerPool...DONE!')
    return worker
