import unittest
from typing import List, Optional, Tuple

import torch

from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import LaneSegmentTrafficLightData
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import (
    convert_feature_layer_to_fixed_size,
    interpolate_points,
)


class TestVectorPreprocessing(unittest.TestCase):
    """Test preprocessing utility functions to assist with builders for vectorized map features."""

    def setUp(self) -> None:
        """Set up test case."""
        self.max_elements = 30
        self.max_points = 20
        self.interpolation = None
        self.traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    def test_interpolate_points_functionality(self) -> None:
        """
        Test interpolating coordinate points.
        """
        coords = torch.tensor([[1, 1], [3, 1], [5, 1]], dtype=torch.float64)

        interpolated_coords = interpolate_points(coords, 5, interpolation='linear')
        self.assertEqual(interpolated_coords.shape, (5, 2))
        torch.testing.assert_allclose(coords, interpolated_coords[::2])
        torch.testing.assert_allclose(interpolated_coords[:, 1], torch.ones((5), dtype=torch.float64))
        self.assertTrue(interpolated_coords[1][0].item() > interpolated_coords[0][0].item())
        self.assertTrue(interpolated_coords[1][0].item() < interpolated_coords[2][0].item())
        self.assertTrue(interpolated_coords[3][0].item() > interpolated_coords[2][0].item())
        self.assertTrue(interpolated_coords[3][0].item() < interpolated_coords[4][0].item())

        interpolated_coords = interpolate_points(coords, 5, interpolation='area')
        self.assertEqual(interpolated_coords.shape, (5, 2))
        torch.testing.assert_allclose(coords, interpolated_coords[::2])
        torch.testing.assert_allclose(interpolated_coords[:, 1], torch.ones((5), dtype=torch.float64))
        self.assertTrue(interpolated_coords[1][0].item() > interpolated_coords[0][0].item())
        self.assertTrue(interpolated_coords[1][0].item() < interpolated_coords[2][0].item())
        self.assertTrue(interpolated_coords[3][0].item() > interpolated_coords[2][0].item())
        self.assertTrue(interpolated_coords[3][0].item() < interpolated_coords[4][0].item())

    def test_interpolate_points_scriptability(self) -> None:
        """
        Tests that the function interpolate_points scripts properly.
        """

        class tmp_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, coords: torch.Tensor, max_points: int, interpolation: str) -> torch.Tensor:
                result = interpolate_points(coords, max_points, interpolation)
                return result

        to_script = tmp_module()
        scripted = torch.jit.script(to_script)

        test_coords = torch.tensor([[1, 1], [3, 1], [5, 1]], dtype=torch.float64)

        py_result = to_script.forward(test_coords, 5, 'linear')
        script_result = scripted.forward(test_coords, 5, 'linear')

        torch.testing.assert_allclose(py_result, script_result)

    def test_convert_feature_layer_to_fixed_size_functionality(self) -> None:
        """
        Test converting variable size data to fixed size tensors.
        """
        coords: List[torch.Tensor] = [torch.tensor([[0.0, 0.0]])]
        traffic_light_data: List[torch.Tensor] = [
            [torch.tensor([LaneSegmentTrafficLightData.encode(TrafficLightStatusType.UNKNOWN)])]
        ]

        coords_tensor, tl_data_tensor, avails_tensor = convert_feature_layer_to_fixed_size(
            coords,
            traffic_light_data,
            self.max_elements,
            self.max_points,
            self.traffic_light_encoding_dim,
            self.interpolation,
        )
        self.assertIsInstance(coords_tensor, torch.DoubleTensor)
        self.assertIsInstance(tl_data_tensor, torch.FloatTensor)
        self.assertIsInstance(avails_tensor, torch.BoolTensor)

        self.assertEqual(coords_tensor.shape, (self.max_elements, self.max_points, 2))
        self.assertEqual(
            tl_data_tensor[0].shape, (self.max_elements, self.max_points, LaneSegmentTrafficLightData.encoding_dim())
        )
        self.assertEqual(avails_tensor.shape, (self.max_elements, self.max_points))

        # test padding without interpolation
        expected_avails = torch.zeros(avails_tensor.shape, dtype=torch.bool)
        expected_avails[0][0] = True
        torch.testing.assert_equal(expected_avails, avails_tensor)

        # test padding with interpolation
        coords_tensor, tl_data_tensor, avails_tensor = convert_feature_layer_to_fixed_size(
            coords,
            traffic_light_data,
            self.max_elements,
            self.max_points,
            self.traffic_light_encoding_dim,
            interpolation='linear',
        )
        expected_avails = torch.zeros(avails_tensor.shape, dtype=torch.bool)
        expected_avails[0][:] = True
        torch.testing.assert_equal(expected_avails, avails_tensor)

    def test_convert_feature_layer_to_fixed_size_scriptability(self) -> None:
        """
        Tests that the function convert_feature_layer_to_fixed_size scripts properly.
        """

        class tmp_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                coords: List[torch.Tensor],
                traffic_light_data: Optional[List[List[torch.Tensor]]],
                max_elements: int,
                max_points: int,
                traffic_light_encoding_dim: int,
                interpolation: Optional[str],
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
                result_coords, result_tl_data, result_avails = convert_feature_layer_to_fixed_size(
                    coords, traffic_light_data, max_elements, max_points, traffic_light_encoding_dim, interpolation
                )
                return result_coords, result_tl_data, result_avails

        to_script = tmp_module()
        scripted = torch.jit.script(to_script)

        test_coords: List[torch.Tensor] = [torch.tensor([[0.0, 0.0]])]
        test_traffic_light_data: List[torch.Tensor] = [
            [torch.tensor([LaneSegmentTrafficLightData.encode(TrafficLightStatusType.UNKNOWN)])]
        ]

        py_result_coords, py_script_result_tl_data, py_script_result_avails = to_script.forward(
            test_coords,
            test_traffic_light_data,
            self.max_elements,
            self.max_points,
            self.traffic_light_encoding_dim,
            self.interpolation,
        )
        script_result_coords, script_result_tl_data, script_result_avails = scripted.forward(
            test_coords,
            test_traffic_light_data,
            self.max_elements,
            self.max_points,
            self.traffic_light_encoding_dim,
            self.interpolation,
        )

        torch.testing.assert_allclose(py_result_coords, script_result_coords)
        torch.testing.assert_allclose(py_script_result_tl_data, script_result_tl_data)
        torch.testing.assert_allclose(py_script_result_avails, script_result_avails)


if __name__ == '__main__':
    unittest.main()
