import logging

from hydra.utils import instantiate
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.script.builders.utils.utils_type import validate_type
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_scenario_filter(cfg: DictConfig) -> ScenarioFilters:
    """
    Builds the scenario filter.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param db: dabatase
    :return: Instance of ScenarioFilters.
    """

    logger.info('Building ScenarioFilter...')
    scenario_filter: ScenarioFilters = instantiate(cfg)
    validate_type(scenario_filter, ScenarioFilters)
    logger.info('Building ScenarioFilter...DONE!')
    return scenario_filter
