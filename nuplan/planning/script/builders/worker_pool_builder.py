import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed

logger = logging.getLogger(__name__)


def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    Builds the worker.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of WorkerPool.
    """
    logger.info('Building WorkerPool...')
    worker: WorkerPool = (
        instantiate(cfg.worker, output_dir=cfg.output_dir)
        if is_target_type(cfg.worker, RayDistributed)
        else instantiate(cfg.worker)
    )
    validate_type(worker, WorkerPool)

    logger.info('Building WorkerPool...DONE!')
    return worker
