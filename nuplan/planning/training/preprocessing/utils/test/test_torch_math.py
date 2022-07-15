import unittest

import torch

from nuplan.planning.training.preprocessing.utils.torch_math import approximate_derivatives_tensor


class TestTorchMath(unittest.TestCase):
    """
    A class to test the functionality of the scriptable torch math library.
    """

    def test_approximate_derivatives_tensor_functionality(self) -> None:
        """
        Tests the numerical accuracy of approximate_derivatives_tensor.
        """
        input_y = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
        input_x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        window_length = 3  # Currently the only supported value
        poly_order = 2
        deriv_order = 1

        expected_output = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.float32)
        actual_output = approximate_derivatives_tensor(input_y, input_x, window_length, poly_order, deriv_order)

        torch.testing.assert_allclose(expected_output, actual_output)

    def test_approximate_derivatives_tensor_scripts_properly(self) -> None:
        """
        Tests that approximate_derivatives_tensor scripts properly.
        """

        class tmp_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                output = approximate_derivatives_tensor(y, x, window_length=3, poly_order=2, deriv_order=1)

                return output

        to_script = tmp_module()
        scripted = torch.jit.script(to_script)

        test_y = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
        test_x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

        py_result = to_script.forward(test_x, test_y)
        script_result = scripted.forward(test_x, test_y)

        torch.testing.assert_allclose(py_result, script_result)


if __name__ == "__main__":
    unittest.main()
