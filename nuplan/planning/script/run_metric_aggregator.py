import logging
import os

import hydra
from omegaconf import DictConfig

from nuplan.planning.script.builders.metric_aggregator_builder import build_metrics_aggregators
from nuplan.planning.script.run_simulation import CONFIG_PATH
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback

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
    assert os.path.exists(cfg.output_dir), f'{cfg.output_dir} does not exist!'

    # Build metric aggregators
    metric_aggregators = build_metrics_aggregators(cfg)

    # Build metric aggregator callback
    metric_save_path = os.path.join(cfg.output_dir, cfg.metric_folder_name)
    metric_aggregator_callback = MetricAggregatorCallback(
        metric_save_path=metric_save_path, metric_aggregators=metric_aggregators
    )
    # Run the aggregator callback
    metric_aggregator_callback.on_run_simulation_end()


if __name__ == '__main__':
    main()
