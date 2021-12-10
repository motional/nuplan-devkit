import logging
from typing import List

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_objectives(cfg: DictConfig) -> List[AbstractObjective]:
    """
    Build objectives based on config
    :param cfg: config
    :return list of objectives.
    """
    instantiated_objectives = []
    for objective_name, objective_type in cfg.objective.items():
        new_objective: AbstractObjective = instantiate(objective_type)
        validate_type(new_objective, AbstractObjective)
        instantiated_objectives.append(new_objective)
    return instantiated_objectives
