from abc import abstractmethod
from typing import Dict, Optional, Type

from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects, TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import sort_dict


class AbstractMLAgents(AbstractObservation):
    """
    Simulate agents based on an ML model.
    """

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario) -> None:
        """
        Initializes the AbstractEgoCentricMLAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        self._model_loader = ModelLoader(model)
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval_us = model.future_trajectory_sampling.step_time * 1e6
        self._num_output_dim = model.future_trajectory_sampling.num_poses

        self._scenario = scenario
        self._ego_anchor_state = scenario.initial_ego_state

        self.step_time = None  # time pass since last simulation iteration
        self._agents: Optional[Dict[str, TrackedObject]] = None

    @abstractmethod
    def _infer_model(self, features: FeaturesType) -> TargetsType:
        """
        Makes a single inference on a Pytorch/Torchscript model.
        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        pass

    @abstractmethod
    def _update_observation_with_predictions(self, agent_predictions: TargetsType) -> None:
        """
        Update smart agent using the predictions from the ML model
        :param agent_predictions: The prediction output from the ML_model
        """
        pass

    def _initialize_agents(self) -> None:
        """
        Initializes the agents based on the first step of the scenario
        """
        unique_agents = {
            tracked_object.track_token: tracked_object
            for tracked_object in self._scenario.initial_tracked_objects.tracked_objects
            if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE
        }
        # TODO: consider agents appearing in the future (not just the first frame)
        self._agents = sort_dict(unique_agents)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._initialize_agents()

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._initialize_agents()
        self._model_loader.initialize()

    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        self.step_time = next_iteration.time_point - iteration.time_point
        self._ego_anchor_state, _ = history.current_state

        # Construct input features
        # TODO: Rename PlannerInitialization to something that also applies to smart agents
        initialization = PlannerInitialization(
            mission_goal=self._scenario.get_mission_goal(),
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            map_api=self._scenario.map_api,
        )
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(next_iteration.index)
        current_input = PlannerInput(next_iteration, history, traffic_light_data)
        features = self._model_loader.build_features(current_input, initialization)

        # Infer model
        predictions = self._infer_model(features)

        # Update observations
        self._update_observation_with_predictions(predictions)

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert self._agents, (
            "ML agent observation has not been initialized!"
            "Please make sure initialize() is called before getting the observation."
        )
        return DetectionsTracks(TrackedObjects(list(self._agents.values())))
