import logging
from typing import List

from hydra.utils import instantiate
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.script.builders.metric_builder import build_metrics_engines
from nuplan.planning.script.builders.planner_builder import build_planner
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallBack
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation import Simulation, SimulationSetup
from nuplan.planning.simulation.simulation_manager.abstract_simulation_manager import AbstractSimulationManager
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)


def build_simulations(cfg: DictConfig,
                      scenario_builder: AbstractScenarioBuilder,
                      worker: WorkerPool,
                      callbacks: List[AbstractCallback],
                      planners: List[AbstractPlanner]) -> List[Simulation]:
    """
    Build simulations.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param scenario_builder: Scenario builder used to extract scenarios
    :param callbacks: Callbacks for simulation.
    :param worker: Worker for job execution
    :param planners: List of pre-built planners to run in simulation.
    :return A dict of simulation engines with challenge names.
    """

    logger.info("Building simulations...")

    # Create Simulation object container
    simulations = list()

    # Retrieve scenarios
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    logger.info("Extracting scenarios...")
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
    logger.info("Extracting scenarios...DONE!")

    logger.info("Building metric engines...")
    metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)
    logger.info("Building metric engines...DONE")

    logger.info(f"Building simulations from {len(planners)} planners and {len(scenarios)} scenarios...")
    for planner in planners:
        for scenario in scenarios:
            # Ego Controller
            ego_controller: AbstractEgoController = instantiate(cfg.ego_controller, scenario=scenario)

            # Simulation Manager
            simulation_manager: AbstractSimulationManager = instantiate(cfg.simulation_manager, scenario=scenario)

            # Perception
            observations: AbstractObservation = instantiate(cfg.observation, scenario=scenario)

            # Metric Engine
            metric_engine = metric_engines_map[scenario.scenario_type]

            extra_callbacks = [
                MetricCallBack(metric_engine=metric_engine, scenario_name=scenario.scenario_name)
            ]
            simulation_callbacks = callbacks + extra_callbacks

            simulation_setup = SimulationSetup(simulation_manager=simulation_manager,
                                               observations=observations,
                                               ego_controller=ego_controller,
                                               scenario=scenario)

            simulation = Simulation(simulation_setup=simulation_setup,
                                    planner=planner,
                                    callbacks=simulation_callbacks,
                                    enable_progress_bar=cfg.enable_simulation_progress_bar,
                                    simulation_history_buffer_duration=cfg.simulation_history_buffer_duration)

            simulations.append(simulation)

    logger.info("Building simulations...DONE!")
    return simulations
