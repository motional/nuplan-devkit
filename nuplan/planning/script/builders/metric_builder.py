import logging
import pathlib
from typing import Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

logger = logging.getLogger(__name__)


def build_high_level_metric(cfg: DictConfig, base_metrics: Dict[str, AbstractMetricBuilder]) -> AbstractMetricBuilder:
    """
    Build a high level metric.
    :param cfg: High level metric config.
    :param base_metrics: A dict of base metrics.
    :return A high level metric.
    """
    # Make it editable
    OmegaConf.set_struct(cfg, False)
    required_metrics: Dict[str, str] = cfg.pop('required_metrics', {})
    OmegaConf.set_struct(cfg, True)

    metric_params = {}
    for metric_param, metric_name in required_metrics.items():
        metric_params[metric_param] = base_metrics[metric_name]

    return instantiate(cfg, **metric_params)


def build_metrics_engines(cfg: DictConfig, scenarios: List[AbstractScenario]) -> Dict[str, MetricsEngine]:
    """
    Build a metric engine for each different scenario type.
    :param cfg: Config.
    :param scenarios: list of scenarios for which metrics should be build.
    :return Dict of scenario types to metric engines.
    """
    main_save_path = pathlib.Path(cfg.output_dir) / cfg.metric_dir

    # Metrics selected by user
    selected_metrics = cfg.selected_simulation_metrics
    if isinstance(selected_metrics, str):
        selected_metrics = [selected_metrics]

    simulation_metrics = cfg.simulation_metric
    low_level_metrics: DictConfig = simulation_metrics.get('low_level', {})
    high_level_metrics: DictConfig = simulation_metrics.get('high_level', {})

    metric_engines = {}
    for scenario in scenarios:
        # If we already have the engine for the specific scenario type, we can skip it
        if scenario.scenario_type in metric_engines:
            continue
        # Metrics
        metric_engine = MetricsEngine(main_save_path=main_save_path)

        # TODO: Add scope checks
        scenario_type = scenario.scenario_type
        scenario_metrics: DictConfig = simulation_metrics.get(scenario_type, {})
        metrics_in_scope = low_level_metrics.copy()
        metrics_in_scope.update(scenario_metrics)

        high_level_metric_in_scope = high_level_metrics.copy()
        # We either pick the selected metrics if any is specified, or all metrics
        if selected_metrics is not None:
            metrics_in_scope = {
                metric_name: metrics_in_scope[metric_name]
                for metric_name in selected_metrics
                if metric_name in metrics_in_scope
            }
            high_level_metric_in_scope = {
                metric_name: high_level_metrics[metric_name]
                for metric_name in selected_metrics
                if metric_name in high_level_metric_in_scope
            }
        base_metrics = {
            metric_name: instantiate(metric_config) for metric_name, metric_config in metrics_in_scope.items()
        }

        for metric in base_metrics.values():
            metric_engine.add_metric(metric)

        # Add high level metrics
        for metric_name, metric in high_level_metric_in_scope.items():
            high_level_metric = build_high_level_metric(cfg=metric, base_metrics=base_metrics)
            metric_engine.add_metric(high_level_metric)

            # Add the high-level metric to the base metrics, so that other high-level metrics can reuse it
            base_metrics[metric_name] = high_level_metric

        metric_engines[scenario_type] = metric_engine

    return metric_engines
