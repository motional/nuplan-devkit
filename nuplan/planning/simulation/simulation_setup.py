from dataclasses import dataclass

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)


@dataclass
class SimulationSetup:
    """Setup class for contructing a Simulation."""

    time_controller: AbstractSimulationTimeController
    observations: AbstractObservation
    ego_controller: AbstractEgoController
    scenario: AbstractScenario

    def __post_init__(self) -> None:
        """Post-initialization sanity checks."""
        # Other checks
        assert isinstance(
            self.time_controller, AbstractSimulationTimeController
        ), 'Error: simulation_time_controller must inherit from AbstractSimulationTimeController!'
        assert isinstance(
            self.observations, AbstractObservation
        ), 'Error: observations must inherit from AbstractObservation!'
        assert isinstance(
            self.ego_controller, AbstractEgoController
        ), 'Error: ego_controller must inherit from AbstractEgoController!'

    def reset(self) -> None:
        """
        Reset all simulation controllers
        """
        self.observations.reset()
        self.ego_controller.reset()
        self.time_controller.reset()


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
