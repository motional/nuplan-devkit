import logging
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.script.builders.metric_builder import build_metric_categories
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.utils.utils_config import update_config_for_nuboard
from nuplan.planning.script.default_path import set_default_path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/nuboard")
if os.path.basename(CONFIG_PATH) != "nuboard":
    CONFIG_PATH = os.path.join(CONFIG_PATH, "nuboard")
CONFIG_NAME = 'default_nuboard'


def initialize_nuboard(cfg: DictConfig) -> NuBoard:
    """
    Sets up dependencies and instantiates a NuBoard object.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: NuBoard object.
    """

    # Update and override configs for nuboard
    update_config_for_nuboard(cfg=cfg)

    scenario_builder = build_scenario_builder(cfg)
    metric_categories = build_metric_categories(cfg)

    # Build vehicle parameters
    vehicle_parameters: VehicleParameters = instantiate(cfg.vehicle_parameters)
    profiler_path = None
    if cfg.profiler_path:
        profiler_path = Path(cfg.profiler_path)

    nuboard = NuBoard(profiler_path=profiler_path,
                      nuboard_paths=cfg.simulation_path,
                      scenario_builder=scenario_builder,
                      port_number=cfg.port_number,
                      metric_categories=metric_categories,
                      resource_prefix=cfg.resource_prefix,
                      vehicle_parameters=vehicle_parameters)

    return nuboard


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    nuboard = initialize_nuboard(cfg)
    nuboard.run()


if __name__ == '__main__':
    main()
