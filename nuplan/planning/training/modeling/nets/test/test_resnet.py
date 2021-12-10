import unittest

import torch
from nuplan.planning.training.modeling.nets.resnet import ResNet1D


class TestResNet1D(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ResNet1D(3, 10)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works
        """
        self.assertNotEqual(self.model, None)

    def test_output_shape(self) -> None:
        """
        Test final output shape is as expected
        """
        inp = torch.randn(1, 3, 23)
        out = self.model(inp)
        self.assertEqual(out.shape, (1, 10, 23))


if __name__ == "__main__":
    unittest.main()
