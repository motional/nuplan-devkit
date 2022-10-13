import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective

logger = logging.getLogger(__name__)


def build_objectives(cfg: DictConfig) -> List[AbstractObjective]:
    """
    Build objectives based on config
    :param cfg: config
    :return list of objectives.
    """
    instantiated_objectives = []

    scenario_type_loss_weighting = (
        cfg.scenario_type_weights.scenario_type_loss_weights
        if ('scenario_type_weights' in cfg and 'scenario_type_loss_weights' in cfg.scenario_type_weights)
        else {}
    )
    for objective_name, objective_type in cfg.objective.items():
        new_objective: AbstractObjective = instantiate(
            objective_type, scenario_type_loss_weighting=scenario_type_loss_weighting
        )
        validate_type(new_objective, AbstractObjective)
        instantiated_objectives.append(new_objective)
    return instantiated_objectives
