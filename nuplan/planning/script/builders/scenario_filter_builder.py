import logging
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def is_valid_token(token: Any) -> bool:
    """
    Basic check that a scenario token is the right type/length.
    :token: parsed by hydra.
    :return: true if it looks valid, otherwise false.
    """
    if not isinstance(token, str) or len(token) != 16:
        return False

    try:
        return bytearray.fromhex(token).hex() == token
    except (TypeError, ValueError):
        return False


def build_scenario_filter(cfg: DictConfig) -> ScenarioFilter:
    """
    Builds the scenario filter.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param db: dabatase.
    :return: Instance of ScenarioFilter.
    """
    logger.info('Building ScenarioFilter...')
    if cfg.scenario_tokens and not all(map(is_valid_token, cfg.scenario_tokens)):
        raise RuntimeError(
            "Expected all scenario tokens to be 16-character strings. Your shell may strip quotes "
            "causing hydra to parse a token as a float, so consider passing them like "
            "scenario_filter.scenario_tokens='[\"595322e649225137\", ...]'"
        )
    scenario_filter: ScenarioFilter = instantiate(cfg)
    validate_type(scenario_filter, ScenarioFilter)
    logger.info('Building ScenarioFilter...DONE!')
    return scenario_filter
