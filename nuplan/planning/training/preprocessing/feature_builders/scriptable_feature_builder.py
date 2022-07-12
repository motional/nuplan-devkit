from abc import abstractmethod
from typing import Dict, List, Tuple

import torch

from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder


class ScriptableFeatureBuilder(torch.nn.Module, AbstractFeatureBuilder):
    """
    A FeatureBuilder that supports exporting via TorchScript.
    """

    @abstractmethod
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        This method contains the logic that will be exported when scripted.
        :param tensor_data: The input tensor data to the function. This will be computed by the C++ engine as configured
            with `precomputed_feature_config()`
        :param list_tensor_data: The input List[tensor] data to the function. This will be computed by the C++ engine
            as configured with `precomputed_feature_config()`
        :param list_list_tensor_data: The input List[List[tensor]] data to the function. This will be computed by the C++ engine
            as configured with `precomputed_feature_config()`
        :return: The output from the function.
        """
        raise NotImplementedError()

    @abstractmethod
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Export the config used for the precomputed features.
        This method will be scripted and used by the C++ engine to determine which features will be computed as input to `scriptable_forward()`.
        :return: The config required for precomputed features. This is a dict of [method_name -> [method_parameter, parameter_value]]
        """
        raise NotImplementedError()
