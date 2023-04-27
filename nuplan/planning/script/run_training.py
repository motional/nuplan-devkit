import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from nuplan.planning.training.experiments.training import TrainingEngine, build_training_engine

logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config/training')

if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
    CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != 'training':
    CONFIG_PATH = os.path.join(CONFIG_PATH, 'training')
CONFIG_NAME = 'default_training'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    if cfg.py_func == 'train':
        # Build training engine
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info('Starting training...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'test':
        # Build training engine
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"):
            engine = build_training_engine(cfg, worker)

        # Test model
        logger.info('Starting testing...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'cache':
        # Precompute and cache all features
        logger.info('Starting caching...')
        if cfg.worker == "ray_distributed" and cfg.worker.use_distributed:
            raise AssertionError("ray in distributed mode will not work with this job")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')


if __name__ == '__main__':
    main()
