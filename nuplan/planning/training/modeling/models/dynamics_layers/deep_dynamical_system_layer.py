import torch
import torch.nn as nn

from nuplan.planning.training.modeling.models.dynamics_layers.abstract_dynamics import DynamicsLayer


class DeepDynamicalSystemLayer(nn.Module):
    """
    Class to forward simulate a dynamical systems
    for k steps, given an initial condition and
    inputs for each k step.

    By subclassing nn.Module, it can be integrated
    in a pipeline where gradient-based optimization
    is employed.

    Adapted from https://arxiv.org/abs/1908.00219 (Eq.ns 6 in
    the paper have slightly different kinematics)
    """

    def __init__(self, dynamics: DynamicsLayer) -> None:
        """
        Class constructor.
        """
        super().__init__()
        self.dynamics = dynamics

    def forward(
        self,
        initial_state: torch.FloatTensor,
        controls: torch.FloatTensor,
        timestep: float,
        agents_pars: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass.
        Returns state at each time step k

        :param initial_state: torch.FloatTensor [..., dynamics.state_dim()]
        :param controls: torch.FloatTensor[..., k, dynamics.control_dim()]
        :param timestep: float
        :param agents_pars: torch.FloatTensor[..., 1/2]   (length, width (optional) )

        :return: state: torch.FloatTensor[..., k, dynamics.state_dim()]
        """
        # Dimensions check
        if initial_state.shape[-1] != self.dynamics.state_dim():
            raise RuntimeError(f'State dimension must be {self.dynamics.state_dim()}, got {initial_state.shape[-1]}')
        if controls.shape[-1] != self.dynamics.input_dim():
            raise RuntimeError(f'Control dimension must be {self.dynamics.input_dim()}, got {controls.shape[-1]}')

        xout = torch.empty(
            (*controls.shape[:-1], self.dynamics.state_dim()), dtype=initial_state.dtype, device=initial_state.device
        )
        for i in range(controls.shape[-2]):
            initial_state = self.dynamics.forward(initial_state, controls[..., i, :], timestep, agents_pars)
            xout[..., i, :] = initial_state

        return xout
