from abc import abstractmethod
from typing import Dict, List, Tuple

import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


class ScriptableTorchModuleWrapper(TorchModuleWrapper):
    """
    An interface representing a model that can be exported with TorchScript
    """

    def __init__(
        self,
        future_trajectory_sampling: TrajectorySampling,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ):
        """
        Construct a scriptable model with feature and target builders.
        :param future_trajectory_sampling: Parameters for a predicted trajectory.
        :param feature_builders: The list of builders which will compute features for this model.
        :param target_builders: The list of builders which will compute targets for this model.
        """
        super().__init__(
            future_trajectory_sampling=future_trajectory_sampling,
            feature_builders=feature_builders,
            target_builders=target_builders,
        )

    @abstractmethod
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        This method contains the logic that will be exported when scripted.
        It is expected that the input dictionaries contain the data as created by the supplied feature builders.
        :param tensor_data: The input tensor data to the function.
            This will come from the `scriptable_forward` methods in the provided feature builders.
        :param list_tensor_data: The input List[tensor] data to the function.
            This will come from the `scriptable_forward` methods in the provided feature builders.
        :param list_list_tensor_data: The input List[List[tensor]] data to the function.
            This will come from the `scriptable_forward` methods in the provided feature builders.
        :return: The output from the function.
        """
        raise NotImplementedError()
