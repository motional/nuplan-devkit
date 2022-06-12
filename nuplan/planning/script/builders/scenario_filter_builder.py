import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_scenario_filter(cfg: DictConfig) -> ScenarioFilter:
    """
    Builds the scenario filter.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param db: dabatase.
    :return: Instance of ScenarioFilter.
    """
    logger.info('Building ScenarioFilter...')
    scenario_filter: ScenarioFilter = instantiate(cfg)
    validate_type(scenario_filter, ScenarioFilter)
    logger.info('Building ScenarioFilter...DONE!')
    return scenario_filter
