import unittest

import torch

from nuplan.planning.training.modeling.models.dynamics_layers.abstract_dynamics import DynamicsLayer
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import (
    DeepDynamicalSystemLayer,
)
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle,
)
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_unicycle_layer_rear_axle import (
    KinematicUnicycleLayerRearAxle,
)


def kinematic_bicycle_rear_axle_manual_grad(
    acceleration: float, steering_angle: float, t: float, x0: torch.Tensor, wheelbase: torch.Tensor
) -> torch.Tensor:
    """
    Helper function to manually compute gradient.
    """
    man_grad = torch.zeros(6, 2)
    v0 = torch.sqrt(x0[3] ** 2 + x0[4] ** 2)

    man_grad[3, 0] = t * torch.cos(x0[2])
    man_grad[4, 0] = t * torch.sin(x0[2])
    man_grad[5, 1] = v0 / torch.cos(torch.as_tensor(steering_angle)) ** 2 / wheelbase

    return man_grad


def kinematic_unicycle_rear_axle_manual_grad(curvature: float, jerk: float, t: float, x0: torch.Tensor) -> torch.Tensor:
    """
    Helper function to manually compute gradient.
    """
    man_grad = torch.zeros(7, 2)
    v0 = torch.sqrt(x0[3] ** 2 + x0[4] ** 2)

    man_grad[2, 0] = t * v0
    man_grad[5, 1] = t * torch.cos(x0[2])
    man_grad[6, 1] = t * torch.sin(x0[2])

    return man_grad


class TestDeepKinematicBicycleLayer(unittest.TestCase):
    """
    Test Deep Kinematic Layer.
    """

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.dev = torch.device("cpu")
        self.dtype = torch.float
        self.layer = DeepDynamicalSystemLayer(dynamics=KinematicBicycleLayerRearAxle())
        self.layer.to(device=self.dev)

        self.wheelbase = torch.tensor([2.5], device=self.dev, dtype=self.dtype)
        self.timestep = 0.5
        self.x0 = torch.tensor([0.1, -0.2, 0.0, 0.7, -0.4, 0.1], device=self.dev, dtype=self.dtype)

    def assert_gradient_element_almost_equal(
        self, x0: torch.Tensor, control: torch.Tensor, manual_grad: torch.Tensor, el: int, control_idx: int
    ) -> None:
        """
        Auxiliary function to ensure single element
        of the gradient is computed correctly.
        """
        ctrl = control.clone().detach().requires_grad_(True)
        xnext = self.layer.forward(x0, ctrl, self.timestep, self.wheelbase)
        xnext[0, el].backward()

        self.assertAlmostEqual(ctrl.grad.detach().cpu()[control_idx, 0].item(), manual_grad[el, 0].item())
        self.assertAlmostEqual(ctrl.grad.detach().cpu()[control_idx, 1].item(), manual_grad[el, 1].item())

    def test_autograd_computation_one_step(self) -> None:
        """
        Test autograd calculations for DeepKinematicLayer (one step prediction)
        """
        acceleration = 0.75
        steering_angle = 0.3

        # manual grad computation
        manual_grad = kinematic_bicycle_rear_axle_manual_grad(
            acceleration, steering_angle, self.timestep, self.x0, self.wheelbase
        )

        # element-wise autograd computation and check
        control = (
            torch.tensor([acceleration, steering_angle], device=self.dev, dtype=self.dtype)
            .reshape(1, -1)
            .requires_grad_(True)
        )
        for i in range(self.layer.dynamics.state_dim()):
            self.assert_gradient_element_almost_equal(self.x0, control, manual_grad, i, 0)

    def test_autograd_computation_no_future_leak(self) -> None:
        """
        Check states' gradient does not depend
        on future inputs
        """
        acceleration = 0.75
        steering_angle = 0.3

        # element-wise autograd computation and check
        control = torch.tensor(
            [[acceleration, steering_angle], [acceleration, -steering_angle]], device=self.dev, dtype=self.dtype
        ).requires_grad_(True)

        for i in range(self.layer.dynamics.state_dim()):
            for k in range(control.shape[-2]):
                for j in range(k):
                    ctrl = control.clone().detach().requires_grad_(True)
                    xnext = self.layer.forward(self.x0, ctrl, self.timestep, self.wheelbase)
                    xnext[j, i].backward()

                    self.assertAlmostEqual(ctrl.grad.detach().cpu()[k, 0].item(), 0.0)
                    self.assertAlmostEqual(ctrl.grad.detach().cpu()[k, 1].item(), 0.0)

    class MockDynamics(DynamicsLayer):
        """Mock dynamics for testing forward pass"""

        def forward(
            self,
            initial_state: torch.FloatTensor,
            controls: torch.FloatTensor,
            timestep: float,
            vehicle_parameters: torch.FloatTensor,
        ) -> torch.FloatTensor:
            """Dummy forward pass"""
            x = initial_state[..., 0] + controls[..., 0]
            y = initial_state[..., 1] * controls[..., 0]
            return torch.stack((x, y), dim=-1)

        @staticmethod
        def state_dim() -> int:
            """State dim"""
            return 2

        @staticmethod
        def input_dim() -> int:
            """Input dim"""
            return 1

    def test_forward_pass_mock_layer(self) -> None:
        """
        Check forward pass by employing
        a dummy dynamics layer
        """
        mockdynamics = self.MockDynamics()
        layer = DeepDynamicalSystemLayer(dynamics=mockdynamics)
        x0 = torch.tensor([0.0, -1.0], device=self.dev, dtype=self.dtype)
        control = torch.tensor([[2.0], [-1.0]], device=self.dev, dtype=self.dtype)

        # Run mock dynamics through DeepDynamicalSystemLayer's layer forward pass
        xfinal = layer.forward(x0, control, float('nan'), torch.tensor([]))

        # Manually run mock dynamics
        xexp = mockdynamics.forward(
            mockdynamics.forward(x0, control[0, :], float('nan'), torch.tensor([])),
            control[1, :],
            float('nan'),
            torch.tensor([]),
        )

        # Check return values
        for i in range(layer.dynamics.state_dim()):
            self.assertAlmostEqual(xfinal[-1, i].item(), xexp[i].item())

    def test_forward_pass_throws(self) -> None:
        """
        Check forward throws when using
        input with wrong size
        """
        mockdynamics = self.MockDynamics()
        layer = DeepDynamicalSystemLayer(dynamics=mockdynamics)

        # Wrong state
        x0 = torch.tensor([0.0, -1.0, 1.0], device=self.dev, dtype=self.dtype)
        control = torch.tensor([[2.0], [-1.0]], device=self.dev, dtype=self.dtype)
        with self.assertRaises(RuntimeError):
            layer.forward(x0, control, float('nan'), torch.tensor([]))

        # Wrong control
        x0 = torch.tensor([0.0, -1.0], device=self.dev, dtype=self.dtype)
        control = torch.tensor([[2.0, 1.0], [-1.0, 2.0]], device=self.dev, dtype=self.dtype)
        with self.assertRaises(RuntimeError):
            layer.forward(x0, control, float('nan'), torch.tensor([]))

        # Wrong state and control
        x0 = torch.tensor([0.0, -1.0, 1.0], device=self.dev, dtype=self.dtype)
        with self.assertRaises(RuntimeError):
            layer.forward(x0, control, float('nan'), torch.tensor([]))


class TestDeepKinematicUnicycleLayer(unittest.TestCase):
    """
    Test Deep Kinematic Unicycle Layer.
    """

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.dev = torch.device("cpu")
        self.dtype = torch.float
        self.layer = DeepDynamicalSystemLayer(dynamics=KinematicUnicycleLayerRearAxle())
        self.layer.to(device=self.dev)

        self.timestep = 0.5
        self.x0 = torch.tensor([0.1, -0.2, 0.0, 0.7, -0.4, 0.75, 0.0], device=self.dev, dtype=self.dtype)

    def assert_gradient_element_almost_equal(
        self, x0: torch.Tensor, control: torch.Tensor, manual_grad: torch.Tensor, el: int, control_idx: int
    ) -> None:
        """
        Auxiliary function to ensure single element
        of the gradient is computed correctly.
        """
        ctrl = control.clone().detach().requires_grad_(True)
        xnext = self.layer.forward(x0, ctrl, self.timestep, None)
        xnext[0, el].backward()

        self.assertAlmostEqual(ctrl.grad.detach().cpu()[control_idx, 0].item(), manual_grad[el, 0].item())
        self.assertAlmostEqual(ctrl.grad.detach().cpu()[control_idx, 1].item(), manual_grad[el, 1].item())

    def test_autograd_computation_one_step(self) -> None:
        """
        Test autograd calculations for DeepKinematicLayer (one step prediction)
        """
        curvature = 0.125
        jerk = 0.3

        # manual grad computation
        manual_grad = kinematic_unicycle_rear_axle_manual_grad(curvature, jerk, self.timestep, self.x0)

        # element-wise autograd computation and check
        control = torch.tensor([curvature, jerk], device=self.dev, dtype=self.dtype).reshape(1, -1).requires_grad_(True)
        for i in range(self.layer.dynamics.state_dim()):
            self.assert_gradient_element_almost_equal(self.x0, control, manual_grad, i, 0)

    def test_autograd_computation_no_future_leak(self) -> None:
        """
        Check states' gradient does not depend
        on future inputs
        """
        curvature = 0.125
        jerk = 0.3

        # element-wise autograd computation and check
        control = torch.tensor(
            [[curvature, jerk], [curvature, -jerk]], device=self.dev, dtype=self.dtype
        ).requires_grad_(True)

        for i in range(self.layer.dynamics.state_dim()):
            for k in range(control.shape[-2]):
                for j in range(k):
                    ctrl = control.clone().detach().requires_grad_(True)
                    xnext = self.layer.forward(self.x0, ctrl, self.timestep, None)
                    xnext[j, i].backward()

                    self.assertAlmostEqual(ctrl.grad.detach().cpu()[k, 0].item(), 0.0)
                    self.assertAlmostEqual(ctrl.grad.detach().cpu()[k, 1].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
