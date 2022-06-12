from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import Observation, Sensors
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class LidarPcObservation(AbstractObservation):
    """
    Replay lidar pointclouds from the scenario.
    """

    def __init__(self, scenario: AbstractScenario):
        """
        Constructor for LidarPcObservation.
        :param scenario: to get observations from.
        """
        self.scenario = scenario
        self.current_iteration = 0

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return Sensors  # type: ignore

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_observation(self) -> Sensors:
        """Inherited, see superclass."""
        return self.scenario.get_sensors_at_iteration(self.current_iteration)

    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
