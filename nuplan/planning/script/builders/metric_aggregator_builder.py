import logging
from pathlib import Path
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.metrics.aggregator.abstract_metric_aggregator import AbstractMetricAggregator

logger = logging.getLogger(__name__)


def build_metrics_aggregators(cfg: DictConfig) -> List[AbstractMetricAggregator]:
    """
    Build a list of metric aggregators.
    :param cfg: Config
    :return A list of metric aggregators, and the path in which they will  save the results
    """
    metric_aggregators = []
    metric_aggregator_configs = cfg.metric_aggregator
    aggregator_save_path = Path(cfg.aggregator_save_path)
    if not is_s3_path(aggregator_save_path):
        aggregator_save_path.mkdir(exist_ok=True, parents=True)
    for metric_aggregator_config_name, metric_aggregator_config in metric_aggregator_configs.items():
        metric_aggregators.append(instantiate(metric_aggregator_config, aggregator_save_path=aggregator_save_path))

    return metric_aggregators
