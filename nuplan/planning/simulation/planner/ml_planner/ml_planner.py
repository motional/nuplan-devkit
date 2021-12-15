from typing import Optional, Type, cast

import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import Detections, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory
from nuplan.planning.training.modeling.nn_model import NNModule
from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import FeatureBuilderMetaData
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class MLPlanner(AbstractPlanner):
    """
    Implements abstract planner interface.
    Used for simulating any ML planner trained through the nuPlan training framework.
    """

    def __init__(self, model: NNModule) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """

        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval = model.future_trajectory_sampling.step_time
        self._num_output_dim = model.future_trajectory_sampling.num_poses

        self._model = model
        self._feature_builders = model.get_list_of_required_feature()

        self._expert_goal_state: Optional[StateSE2] = None
        self._mission_goal: Optional[StateSE2] = None
        self._map_name: Optional[str] = None
        self._map_api: Optional[AbstractMap] = None

    def _initialize_torch(self) -> None:
        """
        Sets up torch/torchscript for inference.
        """
        # We need only inference
        torch.set_grad_enabled(False)

        # Move to device
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _initialize_model(self) -> None:
        """
        Sets up the model.
        """
        self._model.eval()
        self._model = self._model.to(self._device)

    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        # Propagate model
        predictions = self._model.forward(features)

        # Extract trajectory prediction
        trajectory_predicted = cast(Trajectory, predictions['trajectory'])
        trajectory_tensor = trajectory_predicted.data
        trajectory = trajectory_tensor.cpu().detach().numpy()[0]  # retrive first (and only) batch as a numpy array

        # Unravel to expected (N, K) shape
        trajectory = trajectory.reshape(-1, trajectory_predicted.state_size())

        return cast(npt.NDArray[np.float32], trajectory)

    def initialize(self,
                   expert_goal_state: StateSE2,
                   mission_goal: StateSE2,
                   map_name: str,
                   map_api: AbstractMap) -> None:
        """Inherited, see superclass."""
        self._initialize_torch()
        self._initialize_model()

        self._expert_goal_state = expert_goal_state
        self._mission_goal = mission_goal
        self._map_name = map_name
        self._map_api = map_api

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return Detections  # type: ignore

    def compute_trajectory(self, iteration: SimulationIteration,
                           history: SimulationHistoryBuffer) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        # Construct input features
        meta_data = FeatureBuilderMetaData(self._map_api, self._mission_goal, self._expert_goal_state)
        build_args = [history.ego_states, history.observations, meta_data]
        features = {builder.get_feature_unique_name(): builder.get_features_from_simulation(*build_args)
                    for builder in self._feature_builders}
        features = {name: feature.to_feature_tensor() for name, feature in features.items()}
        features = {name: feature.to_device(self._device) for name, feature in features.items()}
        features = {name: feature.collate([feature]) for name, feature in features.items()}

        # Infer model
        predictions = self._infer_model(features)

        # Convert relative poses to absolute states and wrap in a trajectory object.
        anchor_ego_state = history.ego_states[-1]
        states = transform_predictions_to_states(
            predictions, anchor_ego_state, self._future_horizon, self._step_interval)
        trajectory = InterpolatedTrajectory(states)

        return trajectory
