import logging
import pathlib
import time

from nuplan.planning.nuboard.base.data_class import NuBoardFile
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_training_experiment_folder(cfg: DictConfig) -> None:
    """
    Builds the main experiment folder for training.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    logger.info("Building experiment folders...")
    main_exp_folder = pathlib.Path(cfg.output_dir)
    logger.info(f"Experimental folder: {main_exp_folder}")
    main_exp_folder.mkdir(parents=True, exist_ok=True)


def build_simulation_experiment_folder(cfg: DictConfig) -> str:
    """
    Builds the main experiment folder for simulation.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: The main experiment folder path.
    """
    logger.info("Building experiment folders...")
    main_exp_folder = pathlib.Path(cfg.output_dir)
    logger.info(f"\n\n\tFolder where all results are stored: {main_exp_folder}\n")
    main_exp_folder.mkdir(parents=True, exist_ok=True)

    # Build nuboard event file.
    nuboard_filename = main_exp_folder / (f'nuboard_{int(time.time())}' + NuBoardFile.extension())
    nuboard_file = NuBoardFile(
        main_path=cfg.output_dir,
        simulation_folder=cfg.callback.serialization_callback.folder_name,
        metric_folder=cfg.metric_dir
    )

    nuboard_file.save_nuboard_file(file=nuboard_filename)
    logger.info("Building experiment folders...DONE!")

    return main_exp_folder.name
