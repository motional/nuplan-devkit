from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision
from numpy import ndarray
from torch import Tensor

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, FeatureDataType


@dataclass
class Raster(AbstractModelFeature):
    """
    Dataclass that holds map/environment signals in a raster (HxWxC) or (CxHxW) to be consumed by the model.

    :param ego_layer: raster layer that represents the ego's position and extent
    :param agents_layer: raster layer that represents the position and extent of agents surrounding the ego
    :param roadmap_layer: raster layer that represents map information around the ego
    """

    data: FeatureDataType

    def __post_init__(self) -> None:
        """Sanitize attributes of dataclass."""
        self.num_map_channels = 2  # The number of map related channels (roadmap + baseline path)
        # We assume the map related layers are the bottom, and the number of ego and agent layers (with
        # history frames) will be equal. The separation index between ego and agent layers will be
        # (num_total_channels - num_map_channels)//2
        self.ego_agent_sep_channel_num = int((self.num_channels() - self.num_map_channels) // 2)
        shape = self.data.shape
        array_dims = len(shape)
        if (array_dims != 3) and (array_dims != 4):
            raise RuntimeError(f'Invalid raster array. Expected 3 or 4 dims, got {array_dims}.')

    @property
    def num_batches(self) -> Optional[int]:
        """Number of batches in the feature."""
        return None if len(self.data.shape) < 4 else self.data.shape[0]

    def to_feature_tensor(self) -> AbstractModelFeature:
        """Implemented. See interface."""
        to_tensor_torchvision = torchvision.transforms.ToTensor()
        return Raster(data=to_tensor_torchvision(np.asarray(self.data)))

    def to_device(self, device: torch.device) -> Raster:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return Raster(data=self.data.to(device=device))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Raster:
        """Implemented. See interface."""
        return Raster(data=data["data"])

    def unpack(self) -> List[Raster]:
        """Implemented. See interface."""
        return [Raster(data[None]) for data in self.data]

    @staticmethod
    def from_feature_tensor(tensor: torch.Tensor) -> Raster:
        """Implemented. See interface."""
        array = tensor.numpy()

        # So can assume that the torch tensor will always be channels first
        # and the numpy array will always be channels last.
        # So this moves the channels to last when reading the torch tensor
        if len(array.shape) == 4:
            array = array.transpose(0, 2, 3, 1)
        else:
            array = array.transpose(1, 2, 0)

        return Raster(array)

    @property
    def width(self) -> int:
        """
        :return: the width of a raster
        """
        return self.data.shape[-2] if self._is_channels_last() else self.data.shape[-1]  # type: ignore

    @property
    def height(self) -> int:
        """
        :return: the height of a raster
        """
        return self.data.shape[-3] if self._is_channels_last() else self.data.shape[-2]  # type: ignore

    def num_channels(self) -> int:
        """
        Number of raster channels.
        """
        return self.data.shape[-1] if self._is_channels_last() else self.data.shape[-3]  # type: ignore

    @property
    def ego_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the ego layer
        located at channel 0.
        """
        return self._get_data_channel(range(0, self.ego_agent_sep_channel_num))

    @property
    def agents_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the agents layer
        located at channel 1.
        """
        start_channel = self.ego_agent_sep_channel_num
        end_channel = self.num_channels() - self.num_map_channels
        return self._get_data_channel(range(start_channel, end_channel))

    @property
    def roadmap_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the map layer
        located at channel 2.
        """
        return self._get_data_channel(-2)

    @property
    def baseline_paths_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the baseline paths layer
        located at channel 3.
        """
        return self._get_data_channel(-1)

    def _is_channels_last(self) -> bool:
        """
        Check location of channel dimension
        :return True if position [-1] is the number of channels
        """
        # For tensor, channel is put right before the spatial dimention.
        if isinstance(self.data, Tensor):
            return False

        # The default numpy array data format is channel last.
        elif isinstance(self.data, ndarray):
            return True
        else:
            raise RuntimeError(
                f'The data needs to be either numpy array or torch Tensor, but got type(data): {type(self.data)}'
            )

    def _get_data_channel(self, index: Union[int, range]) -> FeatureDataType:
        """
        Extract channel data
        :param index: of layer
        :return: data corresponding to layer
        """
        if self._is_channels_last():
            return self.data[..., index]
        else:
            return self.data[..., index, :, :]
