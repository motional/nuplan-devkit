import unittest
from unittest.mock import Mock, patch

from nuplan.database.nuplan_db.rendering_utils import render_lidar_box
from nuplan.database.tests.nuplan_db_test_utils import (
    get_test_nuplan_db,
    get_test_nuplan_lidar_box,
    get_test_nuplan_lidarpc_with_blob,
)


class TestRendering(unittest.TestCase):
    """These tests don't assert anything, but they will fail if the rendering code throws an exception."""

    def test_lidar_pc_render(self) -> None:
        """Test Lidar PC render."""
        db = get_test_nuplan_db()
        lidar_pc = get_test_nuplan_lidarpc_with_blob()
        lidar_pc.render(db)

    @patch('nuplan.database.nuplan_db.rendering_utils.Axes.imshow', autospec=True)
    @patch('nuplan.database.nuplan_db.models.Image.load_as', autospec=True)
    def test_lidar_box_render_img_found(self, loadas_mock: Mock, axes_mock: Mock) -> None:
        """Test Lidar Box render when the image is found"""
        # Setup
        db = get_test_nuplan_db()
        lidar_box = get_test_nuplan_lidar_box()

        # Call the method under test
        render_lidar_box(lidar_box, db)

        # Assertions
        loadas_mock.assert_called_once()
        axes_mock.assert_called_once()

    @patch('nuplan.database.nuplan_db.rendering_utils.box_in_image', autospec=True)
    def test_lidar_box_render_img_not_found(self, box_in_image_mock: Mock) -> None:
        """Test Lidar Box render in the event that the image is not found"""
        # Setup
        box_in_image_mock.return_value = False
        db = get_test_nuplan_db()
        lidar_box = get_test_nuplan_lidar_box()

        # Should raise an assertion error
        with self.assertRaises(AssertionError):
            # Call the method under test
            render_lidar_box(lidar_box, db)


if __name__ == '__main__':
    unittest.main()
