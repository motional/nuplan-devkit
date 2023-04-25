import logging
import os
import pathlib
import time

from omegaconf import DictConfig

from nuplan.common.utils.io_utils import path_exists, safe_path_to_string
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.nuboard.base.data_class import NuBoardFile

logger = logging.getLogger(__name__)


def build_training_experiment_folder(cfg: DictConfig) -> None:
    """
    Builds the main experiment folder for training.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    logger.info('Building experiment folders...')
    main_exp_folder = pathlib.Path(cfg.output_dir)
    logger.info(f'Experimental folder: {main_exp_folder}')
    main_exp_folder.mkdir(parents=True, exist_ok=True)


def build_simulation_experiment_folder(cfg: DictConfig) -> str:
    """
    Builds the main experiment folder for simulation.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: The main experiment folder path.
    """
    logger.info('Building experiment folders...')

    main_exp_folder = pathlib.Path(cfg.output_dir)
    logger.info(f'\n\n\tFolder where all results are stored: {main_exp_folder}\n')

    if not is_s3_path(main_exp_folder):
        main_exp_folder.mkdir(parents=True, exist_ok=True)

    # simulation_log_main_path should be the main path contains all folders example, 'metric' and 'simulation'.
    if 'simulation_log_main_path' in cfg and cfg.simulation_log_main_path is not None:
        exp_folder = pathlib.Path(cfg.simulation_log_main_path)
        logger.info(f'\n\n\tUsing previous simulation logs: {exp_folder}\n')
        if not path_exists(exp_folder):
            raise FileNotFoundError(f'{exp_folder} does not exist.')
    else:
        exp_folder = main_exp_folder

    if "simulation_log_callback" in cfg.callback:
        simulation_folder = cfg.callback.simulation_log_callback.simulation_log_dir
    else:
        simulation_folder = None

    metric_main_path = main_exp_folder / cfg.metric_dir
    if not is_s3_path(metric_main_path):
        metric_main_path.mkdir(parents=True, exist_ok=True)

    # Only build one nuboard event file on master node
    if int(os.environ.get('NODE_RANK', 0)) == 0:
        nuboard_filename = main_exp_folder / (f'nuboard_{int(time.time())}' + NuBoardFile.extension())
        nuboard_file = NuBoardFile(
            simulation_main_path=safe_path_to_string(exp_folder),
            simulation_folder=simulation_folder,
            metric_main_path=safe_path_to_string(exp_folder),
            metric_folder=cfg.metric_dir,
            aggregator_metric_folder=cfg.aggregator_metric_dir,
        )
        nuboard_file.save_nuboard_file(nuboard_filename)

    logger.info('Building experiment folders...DONE!')

    return exp_folder.name
