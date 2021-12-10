from typing import cast, List

from hydra.utils import instantiate
from nuplan.planning.script.builders.model_builder import build_nn_model
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.training.modeling.planning_module import PlanningModule
from omegaconf import DictConfig, ListConfig, OmegaConf


def build_planner(planner_cfg: DictConfig) -> AbstractPlanner:
    """
    Instantiate planner
    :param planner_cfg: config of a planner
    :return AbstractPlanner
    """
    if is_target_type(planner_cfg, MLPlanner):
        # Build model and feature builders needed to run an ML model in simulation
        nn_model = build_nn_model(planner_cfg.model_config)
        model = PlanningModule.load_from_checkpoint(planner_cfg.checkpoint_path, model=nn_model).model

        # Remove config elements that are redundant to MLPlanner
        config = planner_cfg.copy()
        OmegaConf.set_struct(config, False)
        config.pop('model_config')
        config.pop('checkpoint_path')
        OmegaConf.set_struct(config, True)

        planner: AbstractPlanner = instantiate(config, model=model)
    else:
        planner = cast(AbstractPlanner, instantiate(planner_cfg))

    return planner


def build_planners(planner_cfg: DictConfig) -> List[AbstractPlanner]:
    """
    Instantiate multiple planners by calling build_planner
    :param planner_cfg: config of a planner
    :return planners: List of AbstractPlanners
    """
    planner_cfgs = planner_cfg if isinstance(planner_cfg, ListConfig) else [planner_cfg]
    planners: List[AbstractPlanner] = [build_planner(planner) for planner in planner_cfgs]
    return planners
