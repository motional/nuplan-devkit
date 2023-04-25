from typing import Type

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractMap, MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.predictor.abstract_predictor import (
    AbstractPredictor,
    PredictorInitialization,
    PredictorInput,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


def get_mock_predictor_initialization() -> PredictorInitialization:
    """
    Returns a mock PredictorInitialization for testing.
    :return: PredictorInitialization.
    """
    return PredictorInitialization(MockAbstractMap())


def get_mock_predictor_input(buffer_size: int = 1) -> PredictorInput:
    """
    Returns a mock PredictorInput for testing.
    :return: PredictorInput.
    """
    scenario = MockAbstractScenario()
    history_buffer = SimulationHistoryBuffer.initialize_from_list(
        buffer_size,
        [scenario.initial_ego_state for _ in range(buffer_size)],
        [scenario.initial_tracked_objects for _ in range(buffer_size)],
        0.5,
    )
    return PredictorInput(
        iteration=SimulationIteration(TimePoint(0), 0), history=history_buffer, traffic_light_data=None
    )


class MockAbstractPredictor(AbstractPredictor):
    """
    Mock Predictor class for testing the AbstractPredictor interface
    """

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def initialize(self, initialization: PredictorInitialization) -> None:
        """Inherited, see superclass."""
        self._map_api = initialization.map_api

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_predicted_trajectories(self, current_input: PredictorInput) -> DetectionsTracks:
        """Inherited, see superclass."""
        _, present_observation = current_input.history.current_state
        return present_observation
