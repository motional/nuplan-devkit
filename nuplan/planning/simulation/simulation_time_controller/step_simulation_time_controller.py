from typing import Optional, cast

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class StepSimulationTimeController(AbstractSimulationTimeController):
    """
    Class handling simulation time and completion.
    """

    def __init__(self, scenario: AbstractScenario):
        """
        Initialize simulation control.
        """
        self.current_iteration_index = 0
        self.scenario = scenario

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration_index = 0

    def get_iteration(self) -> SimulationIteration:
        """Inherited, see superclass."""
        scenario_time = self.scenario.get_time_point(self.current_iteration_index)
        return SimulationIteration(time_point=scenario_time, index=self.current_iteration_index)

    def next_iteration(self) -> Optional[SimulationIteration]:
        """Inherited, see superclass."""
        self.current_iteration_index += 1
        return None if self.reached_end() else self.get_iteration()

    def reached_end(self) -> bool:
        """Inherited, see superclass."""
        return self.current_iteration_index >= self.number_of_iterations() - 1

    def number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return cast(int, self.scenario.get_number_of_iterations())
