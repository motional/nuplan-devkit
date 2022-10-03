import torch

from nuplan.planning.training.modeling.models.dynamics_layers.abstract_dynamics import DynamicsLayer
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_unicycle_layers_utils import (
    InputIndex,
    StateIndex,
)


class KinematicUnicycleLayerRearAxle(DynamicsLayer):
    """
    Class to forward simulate a dynamical system
    for 1 step, given an initial condition and
    an input.

    The model is a Kinematic unicycle model
    based on first order Euler discretization.
    Reference point is rear axle of vehicle.
    State is (x, y, yaw, vel_x, vel_y, accel_x, accel_y).
    Input is (curvature, jerk).

    Note: Forward Euler means that the inputs
    at time 0 will affect vel_x, vel_y at time 2 and x, y at time 3.

    By subclassing nn.Module, it can be integrated
    in a pipeline where gradient-based optimization
    is employed.

    Adapted from unicycle model in https://arxiv.org/pdf/2109.13602.pdf,
    itself an adaption of the deep kinematic model of https://arxiv.org/abs/1908.00219.
    """

    def forward(
        self,
        initial_state: torch.FloatTensor,
        controls: torch.FloatTensor,
        timestep: float,
        vehicle_parameters: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Apply (curvature, jerk) to agent to obtain next sampled state.

        Note: when using the sampled state (e.g., with an imitation loss),
        pay particular care to yaw and 0 <-> 2pi transitions.

        Tensors below have ellipses, since they can be, e.g. (for initial_state),:
        - torch.FloatTensor[state_dim()] for a single batch, single vehicle
        - torch.FloatTensor[num_vehicles, state_dim()] for a single batch, num_vehicles vehicles
        - torch.FloatTensor[num_batches, num_vehicles, state_dim()] for num_batches batches, num_vehicles vehicles
        - torch.FloatTensor[num_vehicles, num_batches, state_dim()] for num_batches batches, num_vehicles vehicles

        :param initial_state: torch.FloatTensor[..., KinematicUnicycleLayer.state_dim()]
        :param controls: torch.FloatTensor[..., KinematicUnicycleLayer.control_dim()]
        :param timestep: float
        :param vehicle_parameters: torch.FloatTensor[..., 1/2]   (length, width (optional) )

        :return: state: torch.FloatTensor[..., KinematicUnicycleLayer.state_dim()]
        """
        x = initial_state[..., StateIndex.X_POS] + initial_state[..., StateIndex.X_VELOCITY] * timestep
        y = initial_state[..., StateIndex.Y_POS] + initial_state[..., StateIndex.Y_VELOCITY] * timestep
        vel_init = torch.sqrt(
            torch.square(initial_state[..., StateIndex.X_VELOCITY])
            + torch.square(initial_state[..., StateIndex.Y_VELOCITY])
        )
        yaw = initial_state[..., StateIndex.YAW] + controls[..., InputIndex.CURVATURE] * vel_init * timestep
        vel_x = initial_state[..., StateIndex.X_VELOCITY] + initial_state[..., StateIndex.X_ACCEL] * timestep
        vel_y = initial_state[..., StateIndex.Y_VELOCITY] + initial_state[..., StateIndex.Y_ACCEL] * timestep
        accel_init = torch.sqrt(
            torch.square(initial_state[..., StateIndex.X_ACCEL]) + torch.square(initial_state[..., StateIndex.Y_ACCEL])
        )
        accel = accel_init + controls[..., InputIndex.JERK] * timestep
        accel_x = accel * torch.cos(initial_state[..., StateIndex.YAW])
        accel_y = accel * torch.sin(initial_state[..., StateIndex.YAW])
        return torch.stack((x, y, yaw, vel_x, vel_y, accel_x, accel_y), dim=-1)

    @staticmethod
    def state_dim() -> int:
        """
        Utility function returning state dimension.
        States are (x, y, yaw, vel_x, vel_y, accel_x, accel_y)
        (same as nuplan.training.modeling.models.dynamics_layers.kinematic_unicycle_layers_utils.StateIndex)
        """
        return 7

    @staticmethod
    def input_dim() -> int:
        """
        Utility function returning control dimension.
        Controls are (curvature, jerk)
        (same as nuplan.training.modeling.models.dynamics_layers.kinematic_unicycle_layers_utils.InputIndex)
        """
        return 2
