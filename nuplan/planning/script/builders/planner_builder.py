from typing import List, Optional, Type, cast

from hydra._internal.utils import _locate
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper


def _build_planner(planner_cfg: DictConfig, scenario: Optional[AbstractScenario]) -> AbstractPlanner:
    """
    Instantiate planner
    :param planner_cfg: config of a planner
    :param scenario: scenario
    :return AbstractPlanner
    """
    config = planner_cfg.copy()
    if is_target_type(planner_cfg, MLPlanner):
        # Build model and feature builders needed to run an ML model in simulation
        torch_module_wrapper = build_torch_module_wrapper(planner_cfg.model_config)
        model = LightningModuleWrapper.load_from_checkpoint(
            planner_cfg.checkpoint_path, model=torch_module_wrapper
        ).model

        # Remove config elements that are redundant to MLPlanner
        OmegaConf.set_struct(config, False)
        config.pop('model_config')
        config.pop('checkpoint_path')
        OmegaConf.set_struct(config, True)

        planner: AbstractPlanner = instantiate(config, model=model)
    else:
        planner_cls: Type[AbstractPlanner] = _locate(config._target_)

        if planner_cls.requires_scenario:
            assert scenario is not None, (
                "Scenario was not provided to build the planner. " f"Planner {config} can not be build!"
            )
            planner = cast(AbstractPlanner, instantiate(config, scenario=scenario))
        else:
            planner = cast(AbstractPlanner, instantiate(config))

    return planner


def build_planners(planner_cfg: DictConfig, scenario: Optional[AbstractScenario]) -> List[AbstractPlanner]:
    """
    Instantiate multiple planners by calling build_planner
    :param planners_cfg: planners config
    :param scenario: scenario
    :return planners: List of AbstractPlanners
    """
    return [_build_planner(planner, scenario) for planner in planner_cfg.values()]
