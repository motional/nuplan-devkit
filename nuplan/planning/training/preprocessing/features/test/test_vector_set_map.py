import unittest
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch

from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap


class TestVectorSetMap(unittest.TestCase):
    """Test vector set map feature representation."""

    def setUp(self) -> None:
        """Set up test case."""
        self.coords: Dict[str, List[npt.NDArray[np.float32]]] = {
            'LANE': [np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])],
            'ROUTE': [np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])],
        }

        self.traffic_light_data: Dict[str, List[npt.NDArray[np.int64]]] = {
            'LANE': [np.array([[[0, 0, 0, 1], [1, 0, 0, 0]], [[0, 0, 0, 1], [1, 0, 0, 0]]])]
        }

        self.availabilities: Dict[str, List[npt.NDArray[np.bool_]]] = {
            'LANE': [np.array([[True, True], [True, True]])],
            'ROUTE': [np.array([[True, True, False], [True, True, False]])],
        }

    def test_vector_set_map_feature(self) -> None:
        """
        Test the core functionality of features.
        """
        feature = VectorSetMap(
            coords=self.coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities
        )
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(VectorSetMap.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(list(feature.coords.values())[0][0], np.ndarray)
        self.assertIsInstance(list(feature.traffic_light_data.values())[0][0], np.ndarray)
        self.assertIsInstance(list(feature.availabilities.values())[0][0], np.ndarray)

        feature = feature.to_feature_tensor()
        self.assertIsInstance(list(feature.coords.values())[0][0], torch.Tensor)
        self.assertIsInstance(list(feature.traffic_light_data.values())[0][0], torch.Tensor)
        self.assertIsInstance(list(feature.availabilities.values())[0][0], torch.Tensor)

    def test_feature_layer_mismatch(self) -> None:
        """
        Test when same feature layers not present across feature.
        """
        # traffic light layer not in coords
        coords: Dict[str, List[npt.NDArray[np.float32]]] = {
            'ROUTE': [np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])],
        }
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

        # coords layer not in availabilities
        availabilities: Dict[str, List[npt.NDArray[np.bool_]]] = {
            'LANE': [np.array([[True, True], [True, True]])],
        }
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=self.coords, traffic_light_data=self.traffic_light_data, availabilities=availabilities)

    def test_dimension_mismatch(self) -> None:
        """
        Test when feature dimensions don't match within or across feature layers.
        """
        # mismatching dimensions between coords and tl status/avails
        coords: Dict[str, List[npt.NDArray[np.float32]]] = {
            'LANE': [np.array([[[0.0, 0.0]], [[0.0, 0.0]]])],
            'ROUTE': [np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])],
        }
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

        # inconsistent batch size between coords and tl status/avails
        coords = {
            'LANE': [
                np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]),
                np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]),
            ],
            'ROUTE': [
                np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]),
                np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]),
            ],
        }
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

        # inconsistent batch size across map feature layers
        coords = {
            'LANE': [np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])],
            'ROUTE': [
                np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]),
                np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]),
            ],
        }
        availabilities: Dict[str, List[npt.NDArray[np.bool_]]] = {
            'LANE': [np.array([[True, True], [True, True]])],
            'ROUTE': [
                np.array([[True, True, False], [True, True, False]]),
                np.array([[True, True, False], [True, True, False]]),
            ],
        }
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=availabilities)

    def test_bad_data(self) -> None:
        """
        Test data dimensions are wrong or missing.
        """
        # wrong coords dimensionality
        coords: Dict[str, List[npt.NDArray[np.float32]]] = {
            'LANE': [np.array([[[0.0], [1.0]], [[0.0], [1.0]]])],
            'ROUTE': [np.array([[[0.0], [1.0], [0.0]], [[0.0], [1.0], [0.0]]])],
        }
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

        # empty data
        coords = {
            'LANE': [np.array([])],
            'ROUTE': [np.array([])],
        }
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)


if __name__ == '__main__':
    unittest.main()
