import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric

logger = logging.getLogger(__name__)


def build_training_metrics(cfg: DictConfig) -> List[AbstractTrainingMetric]:
    """
    Build metrics based on config
    :param cfg: config
    :return list of metrics.
    """
    instantiated_metrics = []
    for metric_name, cfg_metric in cfg.training_metric.items():
        new_metric: AbstractTrainingMetric = instantiate(cfg_metric)
        validate_type(new_metric, AbstractTrainingMetric)
        instantiated_metrics.append(new_metric)
    return instantiated_metrics
