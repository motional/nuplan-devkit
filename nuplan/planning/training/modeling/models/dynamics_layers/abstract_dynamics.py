from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DynamicsLayer(ABC, nn.Module):
    """
    Abstract base class for dynamics layers.
    """

    @abstractmethod
    def forward(
        self,
        initial_state: torch.FloatTensor,
        controls: torch.FloatTensor,
        timestep: float,
        vehicle_parameters: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Apply controls to model to obtain next sampled state.

        Tensors below have ellipses, since they can be, e.g. (for initial_state),:
        - torch.FloatTensor[state_dim()] for a single batch, single vehicle
        - torch.FloatTensor[num_vehicles, state_dim()] for a single batch, num_vehicles vehicles
        - torch.FloatTensor[num_batches, num_vehicles, state_dim()] for num_batches batches, num_vehicles vehicles
        - torch.FloatTensor[num_vehicles, num_batches, state_dim()] for num_batches batches, num_vehicles vehicles

        :param initial_state: torch.FloatTensor[..., DynamicsLayer.state_dim()]
        :param controls: torch.FloatTensor[..., DynamicsLayer.control_dim()]
        :param timestep: float
        :param vehicle_parameters: torch.FloatTensor[..., 1/2]   (length, width (optional) )

        :return: state: torch.FloatTensor[..., DynamicsLayer.state_dim()]
        """
        pass

    @staticmethod
    @abstractmethod
    def state_dim() -> int:
        """
        Utility function returning state dimension.
        """
        pass

    @staticmethod
    @abstractmethod
    def input_dim() -> int:
        """
        Utility function returning control dimension.
        """
        pass
