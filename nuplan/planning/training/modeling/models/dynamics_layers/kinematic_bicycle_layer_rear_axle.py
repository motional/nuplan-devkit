import torch

from nuplan.planning.training.modeling.models.dynamics_layers.abstract_dynamics import DynamicsLayer
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layers_utils import (
    InputIndex,
    StateIndex,
)


class KinematicBicycleLayerRearAxle(DynamicsLayer):
    """
    Class to forward simulate a dynamical system
    for 1 step, given an initial condition and
    an input.

    The model is a Kinematic bicycle model
    based on first order Euler discretization.
    Reference point is rear axle of vehicle.
    State is (x, y, yaw, vel_x, vel_y, yaw_rate).
    Input is (acceleration, steering_angle).

    Note: Forward Euler means that the inputs
    at time 0 will affect x,y,yaw at time 2.

    By subclassing nn.Module, it can be integrated
    in a pipeline where gradient-based optimization
    is employed.

    Adapted from https://arxiv.org/abs/1908.00219 (Eq.ns 6 in
    the paper have slightly different kinematics)
    """

    def forward(
        self,
        initial_state: torch.FloatTensor,
        controls: torch.FloatTensor,
        timestep: float,
        vehicle_parameters: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Apply (acceleration, steering_angle) to agent to obtain next sampled state.

        Note: when using the sampled state (e.g., with an imitation loss),
        pay particular care to yaw and 0 <-> 2pi transitions.

        Tensors below have ellipses, since they can be, e.g. (for initial_state),:
        - torch.FloatTensor[state_dim()] for a single batch, single vehicle
        - torch.FloatTensor[num_vehicles, state_dim()] for a single batch, num_vehicles vehicles
        - torch.FloatTensor[num_batches, num_vehicles, state_dim()] for num_batches batches, num_vehicles vehicles
        - torch.FloatTensor[num_vehicles, num_batches, state_dim()] for num_batches batches, num_vehicles vehicles

        :param initial_state: torch.FloatTensor[..., KinematicBicycleLayer.state_dim()]
        :param controls: torch.FloatTensor[..., KinematicBicycleLayer.control_dim()]
        :param timestep: float
        :param vehicle_parameters: torch.FloatTensor[..., 1/2]   (length, width (optional) )

        :return: state: torch.FloatTensor[..., KinematicBicycleLayer.state_dim()]
        """
        wheelbase = vehicle_parameters[..., 0]
        vel_init = torch.sqrt(
            initial_state[..., StateIndex.X_VELOCITY] ** 2 + initial_state[..., StateIndex.Y_VELOCITY] ** 2
        )
        vel = vel_init + controls[..., InputIndex.ACCEL] * timestep
        yaw_rate = vel_init * torch.tan(controls[..., InputIndex.STEERING_ANGLE]) / wheelbase
        yaw = initial_state[..., StateIndex.YAW] + initial_state[..., StateIndex.YAW_RATE] * timestep
        vel_x = vel * torch.cos(initial_state[..., StateIndex.YAW])
        vel_y = vel * torch.sin(initial_state[..., StateIndex.YAW])
        x = initial_state[..., StateIndex.X_POS] + initial_state[..., StateIndex.X_VELOCITY] * timestep
        y = initial_state[..., StateIndex.Y_POS] + initial_state[..., StateIndex.Y_VELOCITY] * timestep
        return torch.stack((x, y, yaw, vel_x, vel_y, yaw_rate), dim=-1)

    @staticmethod
    def state_dim() -> int:
        """
        Utility function returning state dimension.
        States are (x, y, yaw, vel_x, vel_y, yaw_rate)
        (same as nuplan.training.modeling.models.dynamics_layers.kinematic_bicycle_layers_utils.StateIndex)
        """
        return 6

    @staticmethod
    def input_dim() -> int:
        """
        Utility function returning control dimension.
        Controls are (acceleration, steering_angle)
        (same as nuplan.training.modeling.models.dynamics_layers.kinematic_bicycle_layers_utils.InputIndex)
        """
        return 2
