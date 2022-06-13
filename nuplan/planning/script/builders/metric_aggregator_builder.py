import logging
from pathlib import Path
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.metrics.aggregator.abstract_metric_aggregator import AbstractMetricAggregator

logger = logging.getLogger(__name__)


def build_metrics_aggregators(cfg: DictConfig) -> List[AbstractMetricAggregator]:
    """
    Build a list of metric aggregators.
    :param cfg: Config
    :return A list of metric aggregators.
    """
    metric_aggregators = []
    metric_aggregator_configs = cfg.metric_aggregator
    aggregator_save_path = Path(cfg.output_dir) / 'aggregator_metric'
    if not aggregator_save_path.exists():
        aggregator_save_path.mkdir(exist_ok=True, parents=True)
    for metric_aggregator_config_name, metric_aggregator_config in metric_aggregator_configs.items():
        metric_aggregators.append(instantiate(metric_aggregator_config, aggregator_save_path=aggregator_save_path))

    return metric_aggregators
