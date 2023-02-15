import logging
from typing import Dict, Type

from nuplan.common.actor_state.agent import Agent
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.predictor.abstract_predictor import (
    AbstractPredictor,
    PredictorInitialization,
    PredictorInput,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)


class LogFuturePredictor(AbstractPredictor):
    """
    Predictor that wraps grabbing future agent trajectories from scenario and returning as ground truth predicted
    trajectories. Predictions are extracted only for agents in input DetectionsTracks.
    """

    # Inherited property, see superclass.
    requires_scenario: bool = True

    def __init__(self, scenario: AbstractScenario, future_trajectory_sampling: TrajectorySampling):
        """
        Constructor of LogFuturePredictor.
        :param scenario: The scenario the predictor is running on.
        :param future_trajectory_sampling: Sampling parameters for future agent trajectory extraction.
        """
        self._scenario = scenario
        self._future_trajectory_sampling = future_trajectory_sampling

    def initialize(self, initialization: PredictorInitialization) -> None:
        """Inherited, see superclass."""
        pass

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_predicted_trajectories(self, current_input: PredictorInput) -> DetectionsTracks:
        """Inherited, see superclass."""
        iteration = current_input.iteration
        scenario_tracked_objects = self._scenario.get_tracked_objects_at_iteration(
            iteration.index, self._future_trajectory_sampling
        )
        scenario_agent_dict: Dict[str, Agent] = {
            agent.metadata.token: agent
            for agent in scenario_tracked_objects.tracked_objects.get_agents()
            if agent.predictions is not None
        }

        _, curr_detections = current_input.history.current_state
        for agent in curr_detections.tracked_objects.get_agents():
            agent.predictions = (
                scenario_agent_dict[agent.metadata.token].predictions
                if agent.metadata.token in scenario_agent_dict
                else None
            )

        return curr_detections
