import logging
import os
import pathlib
import time
from typing import List, Union

import hydra
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import build_simulation_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.planner_builder import build_planners
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.simulation_builder import build_simulations
from nuplan.planning.script.builders.simulation_callback_builder import build_simulation_callbacks
from nuplan.planning.script.builders.utils.utils_config import update_config_for_simulation
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.default_path import set_default_path
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner import run_planner_through_scenarios
from nuplan.planning.training.callbacks.profile_callback import ProfileCallback
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/simulation")
if os.path.basename(CONFIG_PATH) != "simulation":
    CONFIG_PATH = os.path.join(CONFIG_PATH, "simulation")
CONFIG_NAME = 'default_simulation'


def run_simulation(cfg: DictConfig, planner: Union[AbstractPlanner, List[AbstractPlanner]]) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planner: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners
    """

    # Make sure a planner is specified and that two separate planners are not being specified from both arg and config.
    if 'planner' in cfg.keys():
        raise ValueError("Planner specified via both config and argument. Please only specify one planner.")
    if planner is None:
        raise TypeError("Planner argument is None.")

    start_time = time.perf_counter()

    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Update and override configs for simulation
    update_config_for_simulation(cfg=cfg)

    # Configure logger
    build_logger(cfg)

    # Construct builder
    worker = build_worker(cfg)

    # Create output storage folder
    build_simulation_experiment_folder(cfg=cfg)

    # Simulation Callbacks
    output_dir = pathlib.Path(cfg.output_dir)
    callbacks = build_simulation_callbacks(cfg=cfg, output_dir=output_dir)

    # Create profiler if enabled
    profiler = None
    if cfg.enable_profiling:
        logger.info("Profiler is enabled!")
        profiler = ProfileCallback(output_dir=output_dir)

    if profiler:
        # Profile the simulation construction
        profiler.start_profiler("building_simulation")

    # Build scenario builder
    scenario_builder = build_scenario_builder(cfg=cfg)
    # Construct simulations
    if isinstance(planner, AbstractPlanner):
        planner = [planner]
    simulations = build_simulations(cfg=cfg, callbacks=callbacks, scenario_builder=scenario_builder, worker=worker,
                                    planners=planner)
    assert len(simulations) > 0, 'No scenarios found to simulate!'

    if profiler:
        # Stop simulation construction profiling
        profiler.save_profiler("building_simulation")
        # Start simulation running profiling
        profiler.start_profiler("running_simulation")

    logger.info("Running simulation...")
    run_planner_through_scenarios(simulations=simulations,
                                  worker=worker,
                                  num_gpus=cfg.number_of_gpus_used_for_one_simulation,
                                  num_cpus=cfg.number_of_cpus_used_for_one_simulation,
                                  exit_on_failure=cfg.exit_on_failure)
    logger.info("Finished running simulation!")

    # Save profiler
    if profiler:
        profiler.save_profiler("running_simulation")

    end_time = time.perf_counter()
    elapsed_time_s = end_time - start_time
    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))
    logger.info(f"Simulation duration: {time_str} [HH:MM:SS]")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Calls run_simulation to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """

    # Build planners.
    if 'planner' not in cfg.keys():
        raise KeyError("Planner not specified in config. Please specify a planner using 'planner' field.")
    logger.info("Building planners...")
    planners = build_planners(cfg.planner)
    logger.info("Building planners...DONE!")

    # Remove planner from config to make sure run_simulation does not receive multiple planner specifications.
    OmegaConf.set_struct(cfg, False)
    cfg.pop('planner')
    OmegaConf.set_struct(cfg, True)

    # Execute simulation with preconfigured planner(s).
    run_simulation(cfg=cfg, planner=planners)


if __name__ == '__main__':
    main()
