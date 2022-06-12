from typing import List

import timm
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


class RasterModel(TorchModuleWrapper):
    """
    Wrapper around raster-based CNN model that consumes ego, agent and map data in rasterized format
    and regresses ego's future trajectory.
    """

    def __init__(
        self,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        model_name: str,
        pretrained: bool,
        num_input_channels: int,
        num_features_per_pose: int,
        future_trajectory_sampling: TrajectorySampling,
    ):
        """
        Initialize model.
        :param feature_builders: list of builders for features
        :param target_builders: list of builders for targets
        :param model_name: name of the model (e.g. resnet_50, efficientnet_b3)
        :param pretrained: whether the model will be pretrained
        :param num_input_channels: number of input channel of the raster model.
        :param num_features_per_pose: number of features per single pose
        :param future_trajectory_sampling: parameters of predicted trajectory
        """
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )

        num_output_features = future_trajectory_sampling.num_poses * num_features_per_pose
        self._model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=num_input_channels)
        mlp = torch.nn.Linear(in_features=self._model.num_features, out_features=num_output_features)

        if hasattr(self._model, 'classifier'):
            self._model.classifier = mlp
        elif hasattr(self._model, 'fc'):
            self._model.fc = mlp
        else:
            raise NameError('Expected output layer named "classifier" or "fc" in model')

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "raster": Raster,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        raster: Raster = features["raster"]

        predictions = self._model.forward(raster.data)

        return {"trajectory": Trajectory(data=convert_predictions_to_trajectory(predictions))}
