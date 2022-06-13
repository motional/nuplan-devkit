import logging
from typing import List

from omegaconf import DictConfig

from nuplan.planning.script.builders.metric_builder import build_metrics_engines
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.runner.metric_runner import MetricRunner
from nuplan.planning.simulation.simulation_log import SimulationLog

logger = logging.getLogger(__name__)


def build_metric_runners(cfg: DictConfig, simulation_logs: List[SimulationLog]) -> List[MetricRunner]:
    """
    Build metric runners.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param simulation_logs: A list of simulation logs.
    :return A list of metric runners.
    """
    logger.info('Building metric runners...')

    # Create a list of metric runners
    metric_runners = list()

    # Build a list of scenarios
    logger.info('Extracting scenarios...')
    scenarios = [simulation_log.scenario for simulation_log in simulation_logs]
    logger.info('Extracting scenarios...DONE!')

    logger.info('Building metric engines...')
    metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)
    logger.info('Building metric engines...DONE')

    logger.info(f'Building metric_runner from {len(scenarios)} scenarios...')
    for simulation_log in simulation_logs:

        scenario = simulation_log.scenario
        metric_engine = metric_engines_map.get(scenario.scenario_type, None)
        if not metric_engine:
            raise ValueError(f'{scenario.scenario_type} not found in a metric engine.')

        if not simulation_log:
            raise ValueError(f'{scenario.scenario_name} not found in simulation logs.')

        metric_callback = MetricCallback(metric_engine=metric_engine)
        metric_runner = MetricRunner(simulation_log=simulation_log, metric_callback=metric_callback)
        metric_runners.append(metric_runner)

    logger.info('Building metric runners...DONE!')
    return metric_runners
