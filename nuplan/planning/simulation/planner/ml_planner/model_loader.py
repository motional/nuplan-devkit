import torch

from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class ModelLoader:
    """
    A class to load in an NNModel and prepare pytorch for inference.
    Used for simulating ML planners and smart agents throughout the nuPlan training framework.
    """

    def __init__(self, model: TorchModuleWrapper) -> None:
        """
        Initializes the ModelLoader class.
        :param model: Model to use for inference.
        """
        self.feature_builders = model.get_list_of_required_feature()
        self._model = model
        self._initialized = False

    def _initialize_torch(self) -> None:
        """
        Sets up torch/torchscript for inference.
        """
        # We need only inference
        torch.set_grad_enabled(False)

        # Move to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _initialize_model(self) -> None:
        """
        Sets up the model.
        """
        self._model.eval()
        self._model = self._model.to(self.device)

    def initialize(self) -> None:
        """
        Initializes the ModelLoader
        """
        self._initialize_torch()
        self._initialize_model()
        self._initialized = True

    def build_features(self, current_input: PlannerInput, initialization: PlannerInitialization) -> FeaturesType:
        """
        Makes a single inference on a Pytorch/Torchscript model.
        :param current_input: Iteration specific inputs for building the feature.
        :param initialization: Additional data require for building the feature.
        :return: dictionary of FeaturesType types.
        """
        assert self._initialized, "The model loader has not been initialized!"

        features = {
            builder.get_feature_unique_name(): builder.get_features_from_simulation(current_input, initialization)
            for builder in self.feature_builders
        }
        features = {name: feature.to_feature_tensor() for name, feature in features.items()}
        features = {name: feature.to_device(self.device) for name, feature in features.items()}
        features = {name: feature.collate([feature]) for name, feature in features.items()}
        return features

    def infer(self, features: FeaturesType) -> TargetsType:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: dictionary of target types
        """
        assert self._initialized is True, "The model loader has not been initialized!"
        return self._model.forward(features)
