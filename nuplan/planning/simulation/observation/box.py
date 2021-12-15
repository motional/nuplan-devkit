from typing import Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import Detections, Observation
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration


class BoxObservation(AbstractObservation):
    """
    Replay bounding boxes from samples.
    """

    def __init__(self, scenario: AbstractScenario):
        self.scenario = scenario
        self.current_iteration = 0

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return Detections  # type: ignore

    def get_observation(self) -> Detections:
        """ Inherited, see superclass. """
        return self.scenario.get_detections_at_iteration(self.current_iteration)

    def update_observation(self, iteration: SimulationIteration,
                           next_iteration: SimulationIteration,
                           ego_state: EgoState) -> None:
        """ Inherited, see superclass. """
        self.current_iteration = next_iteration.index
