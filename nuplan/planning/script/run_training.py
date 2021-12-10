import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.default_path import set_default_path
from nuplan.planning.training.experiments.training import TrainingEngine, build_training_engine, cache_data
from omegaconf import DictConfig

logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/training")
if os.path.basename(CONFIG_PATH) != "training":
    CONFIG_PATH = os.path.join(CONFIG_PATH, "training")
CONFIG_NAME = 'default_training'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """

    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Construct builder
    worker = build_worker(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Configure logger
    build_logger(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build scenario builder
    scenario_builder = build_scenario_builder(cfg)

    if cfg.py_func == 'train':
        # Build training engine
        engine = build_training_engine(cfg, worker, scenario_builder)

        # Run training
        logger.info('Starting training...')
        engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == "test":
        # Build training engine
        engine = build_training_engine(cfg, worker, scenario_builder)

        # Test model
        logger.info('Starting testing...')
        engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'cache_data':
        # Precompute and cache all features
        cache_data(cfg=cfg, worker=worker, scenario_builder=scenario_builder)
        return None
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')


if __name__ == '__main__':
    main()
