import logging
from dataclasses import dataclass

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_manager.abstract_simulation_manager import AbstractSimulationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationSetup:
    simulation_manager: AbstractSimulationManager
    observations: AbstractObservation
    ego_controller: AbstractEgoController
    scenario: AbstractScenario

    def __post_init__(self) -> None:
        # Other checks
        assert isinstance(self.simulation_manager, AbstractSimulationManager), \
            'Error: simulation_manager must inherit from AbstractSimulationManager!'
        assert isinstance(self.observations, AbstractObservation), \
            'Error: observations must inherit from AbstractObservation!'
        assert isinstance(self.ego_controller, AbstractEgoController), \
            'Error: ego_controller must inherit from AbstractEgoController!'


def validate_planner_setup(setup: SimulationSetup, planner: AbstractPlanner) -> None:
    """
    Validate planner and simulation setup
    :param setup: Simulation setup
    :param planner: Planner to be used
    @raise ValueError in case simulation setup and planner are not a valid combination
    """
    # Validate the setup
    type_observation_planner = planner.observation_type()
    type_observation = setup.observations.observation_type()

    if type_observation_planner != type_observation:
        raise ValueError(
            "Error: The planner did not receive the right observations:"
            f"{type_observation} != {type_observation_planner} planner."
            f"Planner {type(planner)}, Observation:{type(setup.observations)}"
        )
