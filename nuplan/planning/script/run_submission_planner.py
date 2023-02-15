import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from nuplan.planning.script.utils import set_default_path
from nuplan.submission.submission_planner import SubmissionPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config/simulation')

if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
    CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != 'simulation':
    CONFIG_PATH = os.path.join(CONFIG_PATH, 'simulation')
CONFIG_NAME = 'default_submission_planner'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute submission planner which will listen to the simulation and compute trajectory at request
    :param cfg: Configuration that is used to run the experiment.
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Here is where you need to initialize your planner. You have to use hydra (as in the given example),
    # passing your planner configuration file as an argument
    submission_planner = SubmissionPlanner(planner_config=cfg.planner)
    submission_planner.serve()


if __name__ == '__main__':
    main()
