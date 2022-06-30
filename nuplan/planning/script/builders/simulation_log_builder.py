import logging
from pathlib import Path
from typing import List

from omegaconf import DictConfig

from nuplan.planning.simulation.simulation_log import SimulationLog

logger = logging.getLogger(__name__)


def build_simulation_logs(cfg: DictConfig) -> List[SimulationLog]:
    """
    Build a list of simulation logs.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return A list of simulation logs.
    """
    logger.info('Building simulation logs...')

    # Create Simulation object container
    simulation_logs = []

    # main path where the simulation logs save
    simulation_log_path = Path(cfg.simulation_log_main_path) / cfg.callback.simulation_log_callback.simulation_log_dir

    # Folder structure planner -> scenario_type -> scenario file
    for planner_dir_folder in simulation_log_path.iterdir():
        for scenario_type_folder in planner_dir_folder.iterdir():
            for log_name_folder in scenario_type_folder.iterdir():
                for scenario_name_folder in log_name_folder.iterdir():
                    for scenario_log_file in scenario_name_folder.iterdir():
                        simulation_log = SimulationLog.load_data(file_path=scenario_log_file)
                        simulation_logs.append(simulation_log)

    logger.info(f'Building simulation logs: {len(simulation_logs)}...DONE!')

    return simulation_logs
