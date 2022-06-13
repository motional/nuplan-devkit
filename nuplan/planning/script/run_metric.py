import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from nuplan.planning.script.builders.metric_runner_builder import build_metric_runners
from nuplan.planning.script.builders.simulation_log_builder import build_simulation_logs
from nuplan.planning.script.run_simulation import CONFIG_NAME, CONFIG_PATH
from nuplan.planning.script.utils import run_runners, set_default_path, set_up_common_builder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute metrics with simulation logs only.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    assert cfg.simulation_log_main_path is not None, 'Simulation_log_main_path must be set when running metrics.'

    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    profiler_name = 'building_metrics'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build simulation logs
    simulation_logs = build_simulation_logs(cfg=cfg)

    runners = build_metric_runners(cfg=cfg, simulation_logs=simulation_logs)

    if common_builder.profiler:

        # Stop simulation construction profiling
        common_builder.profiler.save_profiler(profiler_name)

    logger.info('Running metrics...')
    run_runners(runners=runners, common_builder=common_builder, cfg=cfg, profiler_name='running_metrics')
    logger.info('Finished running metrics!')


if __name__ == '__main__':
    main()
