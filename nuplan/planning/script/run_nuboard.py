import logging
import os
from pathlib import Path

import hydra
import nest_asyncio
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.utils.utils_config import update_config_for_nuboard
from nuplan.planning.script.utils import set_default_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config/nuboard')

if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
    CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != 'nuboard':
    CONFIG_PATH = os.path.join(CONFIG_PATH, 'nuboard')
CONFIG_NAME = 'default_nuboard'


nest_asyncio.apply()


def initialize_nuboard(cfg: DictConfig) -> NuBoard:
    """
    Sets up dependencies and instantiates a NuBoard object.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: NuBoard object.
    """
    # Update and override configs for nuboard
    update_config_for_nuboard(cfg=cfg)

    scenario_builder = build_scenario_builder(cfg)

    # Build vehicle parameters
    vehicle_parameters: VehicleParameters = instantiate(cfg.scenario_builder.vehicle_parameters)
    profiler_path = None
    if cfg.profiler_path:
        profiler_path = Path(cfg.profiler_path)

    nuboard = NuBoard(
        profiler_path=profiler_path,
        nuboard_paths=cfg.simulation_path,
        scenario_builder=scenario_builder,
        port_number=cfg.port_number,
        resource_prefix=cfg.resource_prefix,
        vehicle_parameters=vehicle_parameters,
        async_scenario_rendering=cfg.async_scenario_rendering,
        scenario_rendering_frame_rate_cap_hz=cfg.scenario_rendering_frame_rate_cap_hz,
    )

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
