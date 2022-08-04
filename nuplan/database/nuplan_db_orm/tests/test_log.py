import unittest
from unittest.mock import MagicMock, Mock, PropertyMock, patch

from nuplan.database.nuplan_db_orm.log import Log


class TestLog(unittest.TestCase):
    """Test class Log"""

    def setUp(self) -> None:
        """
        Initializes a test Log
        """
        self.log = Log()

    @patch("nuplan.database.nuplan_db_orm.log.inspect", autospec=True)
    def test_session(self, inspect: Mock) -> None:
        """
        Tests _session method
        """
        # Setup
        mock_session = PropertyMock()
        inspect.return_value = Mock()
        inspect.return_value.session = mock_session

        # Call method under test
        result = self.log._session()

        # Assertions
        inspect.assert_called_once_with(self.log)
        mock_session.assert_called_once()
        self.assertEqual(result, mock_session.return_value)

    def test_images(self) -> None:
        """
        Tests images property
        """
        # Setup
        mock_image_1 = Mock()
        mock_image_2 = Mock()

        mock_camera_1 = MagicMock()
        mock_camera_1.images = Mock()
        mock_camera_1.images.__iter__ = Mock(return_value=iter([mock_image_1, mock_image_2]))

        mock_camera_2 = MagicMock()
        mock_camera_2.images = Mock()
        mock_camera_2.images.__iter__ = Mock(return_value=iter([mock_image_2, mock_image_1]))

        self.log.cameras = [mock_camera_1, mock_camera_2]

        # Call method under test
        result = self.log.images

        # Assertions
        self.assertEqual(result, [mock_image_1, mock_image_2, mock_image_2, mock_image_1])

    def test_lidar_pcs(self) -> None:
        """
        Test lidar_pcs property
        """
        # Setup
        mock_lidar_pc_1 = Mock()
        mock_lidar_pc_2 = Mock()

        mock_lidar_1 = MagicMock()
        mock_lidar_1.lidar_pcs = Mock()
        mock_lidar_1.lidar_pcs.__iter__ = Mock(return_value=iter([mock_lidar_pc_1, mock_lidar_pc_2]))

        mock_lidar_2 = MagicMock()
        mock_lidar_2.lidar_pcs = Mock()
        mock_lidar_2.lidar_pcs.__iter__ = Mock(return_value=iter([mock_lidar_pc_2, mock_lidar_pc_1]))

        self.log.lidars = [mock_lidar_1, mock_lidar_2]

        # Call method under test
        result = self.log.lidar_pcs

        # Assertions
        self.assertEqual(result, [mock_lidar_pc_1, mock_lidar_pc_2, mock_lidar_pc_2, mock_lidar_pc_1])

    @patch("nuplan.database.nuplan_db_orm.log.Log.lidar_pcs", autospec=True)
    def test_lidar_boxes(self, mock_lidar_pcs: Mock) -> None:
        """
        Test lidar_boxes method
        """
        # Setup
        mock_lidar_box_1 = Mock()
        mock_lidar_box_2 = Mock()

        mock_lidar_pcs_1 = Mock()
        mock_lidar_pcs_1.lidar_boxes = Mock()
        mock_lidar_pcs_1.lidar_boxes.__iter__ = Mock(return_value=iter([mock_lidar_box_1, mock_lidar_box_2]))

        mock_lidar_pcs_2 = Mock()
        mock_lidar_pcs_2.lidar_boxes = Mock()
        mock_lidar_pcs_2.lidar_boxes.__iter__ = Mock(return_value=iter([mock_lidar_box_2, mock_lidar_box_1]))

        mock_lidar_pcs.__iter__ = Mock(return_value=iter([mock_lidar_pcs_1, mock_lidar_pcs_2]))

        # Call method under test
        result = self.log.lidar_boxes

        # Assertions
        mock_lidar_pcs.__iter__.assert_called()
        self.assertEqual(result, [mock_lidar_box_1, mock_lidar_box_2, mock_lidar_box_2, mock_lidar_box_1])

    @patch("nuplan.database.nuplan_db_orm.log.simple_repr", autospec=True)
    def test_repr(self, simple_repr: Mock) -> None:
        """
        Tests string representation
        """
        # Call method under test
        result = self.log.__repr__()

        # Assertions
        simple_repr.assert_called_once()
        self.assertEqual(result, simple_repr.return_value)


if __name__ == "__main__":
    unittest.main()
