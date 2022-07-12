import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from nuplan.planning.script.builders.metric_aggregator_builder import build_metrics_aggregators
from nuplan.planning.script.run_simulation import CONFIG_PATH
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

CONFIG_NAME = 'default_run_metric_aggregator'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute metric aggregators with the simulation path.
    :param cfg: Hydra config dict.
    """
    metric_save_path = Path(cfg.output_dir) / cfg.metric_folder_name
    if not metric_save_path.exists():
        metric_save_path.mkdir(exist_ok=True, parents=True)

    # Run metric_file integrator if it is set
    if cfg.scenario_metric_paths:
        metric_file_callback = MetricFileCallback(
            scenario_metric_paths=cfg.scenario_metric_paths,
            metric_file_output_path=str(metric_save_path),
            delete_scenario_metric_files=cfg.delete_scenario_metric_files,
        )
        metric_file_callback.on_run_simulation_end()

    # Build metric aggregators
    metric_aggregators = build_metrics_aggregators(cfg)

    # Build metric aggregator callback
    metric_aggregator_callback = MetricAggregatorCallback(
        metric_save_path=str(metric_save_path), metric_aggregators=metric_aggregators
    )
    # Run the aggregator callback
    metric_aggregator_callback.on_run_simulation_end()


if __name__ == '__main__':
    main()
