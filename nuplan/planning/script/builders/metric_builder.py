import logging
import pathlib
from typing import Dict, List

from hydra.utils import instantiate
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_metric_categories(cfg: DictConfig) -> List[str]:

    logger.info("Building metric categories...")
    categories = set()
    for metric_type, metric in cfg.simulation_metric.items():
        for metric_name, metric_config in metric.items():
            categories.add(metric_config.category)
    metric_categories: List[str] = list(categories)

    # Sort ascending
    metric_categories.sort(reverse=False)
    logger.info("Building metric categories...Done!")
    return metric_categories


def build_metrics_engines(cfg: DictConfig, scenarios: List[AbstractScenario]) -> Dict[str, MetricsEngine]:
    """
    Build a metric engine for each differenct scenario type.
    :param cfg: Config
    :param scenarios: list of scenarios for which metrics should be build
    :return Dict of scenario types to metric engines
    """

    main_save_path = pathlib.Path(cfg.output_dir) / cfg.metric_dir

    # Metrics selected by user
    selected_metrics = cfg.selected_simulation_metrics
    if isinstance(selected_metrics, str):
        selected_metrics = [selected_metrics]

    simulation_metrics = cfg.simulation_metric
    common_metrics: DictConfig = simulation_metrics.get('common', {})

    metric_engines = {}
    for scenario in scenarios:
        # If we already have the engine for the specific scenario type, we can skip it
        if scenario.scenario_type in metric_engines:
            continue
        # Metrics
        metric_engine = MetricsEngine(scenario_type=scenario.scenario_type,
                                      main_save_path=main_save_path, timestamp=cfg.experiment_time)

        # TODO: Add scope checks
        scenario_type = scenario.scenario_type
        scenario_metrics: DictConfig = simulation_metrics.get(scenario_type, {})
        metrics_in_scope = common_metrics.copy()
        metrics_in_scope.update(scenario_metrics)

        # We either pick the selected metrics if any is specified, or all metrics
        if selected_metrics is not None:
            metrics_in_scope = {metric_name: metrics_in_scope[metric_name] for metric_name in selected_metrics
                                if metric_name in metrics_in_scope}
        for metric_cfg in metrics_in_scope.values():
            metric_engine.add_metric(instantiate(metric_cfg))

        metric_engines[scenario_type] = metric_engine

    return metric_engines
