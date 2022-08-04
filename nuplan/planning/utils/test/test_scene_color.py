import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from nuplan.planning.utils.color import Color, SceneColor


class TestSceneColor(TestCase):
    """
    Test scene color.
    """

    def setUp(self) -> None:
        """
        Set up.
        """
        self.trajectory_color = MagicMock(name="1", spec=Color)
        self.prediction_bike_color = MagicMock(name="2", spec=Color)
        self.prediction_pedestrian_color = MagicMock(name="3", spec=Color)
        self.prediction_vehicle_color = MagicMock(name="4", spec=Color)

        self.scene_color = SceneColor(
            self.trajectory_color,
            self.prediction_bike_color,
            self.prediction_pedestrian_color,
            self.prediction_vehicle_color,
        )

    def test_init(self) -> None:
        """
        Test initialisation.
        """
        # Assertions
        self.assertEqual(self.scene_color.trajectory_color, self.trajectory_color)
        self.assertEqual(self.scene_color.prediction_bike_color, self.prediction_bike_color)
        self.assertEqual(self.scene_color.prediction_pedestrian_color, self.prediction_pedestrian_color)
        self.assertEqual(self.scene_color.prediction_vehicle_color, self.prediction_vehicle_color)

    def test_iter(self) -> None:
        """
        Tests iteration of colors.
        """
        # Call method under test
        result = [color for color in self.scene_color]

        # Assertions
        self.assertEqual(result[0], self.trajectory_color)
        self.assertEqual(result[1], self.prediction_bike_color)
        self.assertEqual(result[2], self.prediction_pedestrian_color)
        self.assertEqual(result[3], self.prediction_vehicle_color)

    def test_mul(self) -> None:
        """
        Tests multiplication operation.
        """
        # Call method under test
        result = self.scene_color * 0.75

        # Assertions
        self.assertEqual(
            result,
            SceneColor(
                self.trajectory_color.__mul__.return_value,
                self.prediction_bike_color.__mul__.return_value,
                self.prediction_pedestrian_color.__mul__.return_value,
                self.prediction_vehicle_color.__mul__.return_value,
            ),
        )

    @patch("nuplan.planning.utils.color.SceneColor.__mul__")
    def test_rmul(self, mock_mul: Mock) -> None:
        """
        Tests reverse multiplication operation.
        """
        # Call method under test
        result = 0.75 * self.scene_color

        # Assertions
        mock_mul.assert_called_once_with(0.75)
        self.assertEqual(result, mock_mul.return_value)


if __name__ == '__main__':
    unittest.main()
