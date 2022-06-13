from typing import cast

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.utils.utils_type import is_TorchModuleWrapper_config
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper


def build_observations(observation_cfg: DictConfig, scenario: AbstractScenario) -> AbstractObservation:
    """
    Instantiate observations
    :param observation_cfg: config of a planner
    :param scenario: scenario
    :return AbstractObservation
    """
    if is_TorchModuleWrapper_config(observation_cfg):
        # Build model and feature builders needed to run an ML model in simulation
        torch_module_wrapper = build_torch_module_wrapper(observation_cfg.model_config)
        model = LightningModuleWrapper.load_from_checkpoint(
            observation_cfg.checkpoint_path, model=torch_module_wrapper
        ).model

        # Remove config elements that are redundant to MLPlanner
        config = observation_cfg.copy()
        OmegaConf.set_struct(config, False)
        config.pop('model_config')
        config.pop('checkpoint_path')
        OmegaConf.set_struct(config, True)

        observation: AbstractObservation = instantiate(config, model=model, scenario=scenario)
    else:
        observation = cast(AbstractObservation, instantiate(observation_cfg, scenario=scenario))

    return observation
