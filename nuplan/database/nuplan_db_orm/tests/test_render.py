import unittest
from unittest.mock import Mock, patch

from nuplan.database.nuplan_db_orm.rendering_utils import lidar_pc_closest_image, render_lidar_box
from nuplan.database.tests.test_utils_nuplan_db import (
    get_test_nuplan_db,
    get_test_nuplan_lidar_box,
    get_test_nuplan_lidarpc_with_blob,
)


class TestRendering(unittest.TestCase):
    """Some of these tests don't assert anything, but they will fail if the rendering code throws an exception."""

    def setUp(self) -> None:
        """Set up"""
        self.db = get_test_nuplan_db()
        self.lidar_box = get_test_nuplan_lidar_box()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()

    def test_closest_image(self) -> None:
        """Tests the closest_image method"""
        # Call the method under test
        result = lidar_pc_closest_image(self.lidar_pc)

        # Assertions
        # Check if the resulting List is not empty, which it shouldn't be
        self.assertNotEqual(len(result), 0)

    def test_lidar_pc_render(self) -> None:
        """Test Lidar PC render."""
        self.lidar_pc.render(self.db)

    @patch("nuplan.database.nuplan_db_orm.rendering_utils.Axes.imshow", autospec=True)
    @patch("nuplan.database.nuplan_db_orm.image.Image.load_as", autospec=True)
    def test_lidar_box_render_img_found(self, loadas_mock: Mock, axes_mock: Mock) -> None:
        """Test Lidar Box render when the image is found"""
        # Call the method under test
        render_lidar_box(self.lidar_box, self.db)

        # Assertions
        loadas_mock.assert_called_once()
        axes_mock.assert_called_once()

    @patch("nuplan.database.nuplan_db_orm.rendering_utils.box_in_image", autospec=True)
    def test_lidar_box_render_img_not_found(self, box_in_image_mock: Mock) -> None:
        """Test Lidar Box render in the event that the image is not found"""
        # Setup
        box_in_image_mock.return_value = False

        # Should raise an assertion error
        with self.assertRaises(AssertionError):
            # Call the method under test
            render_lidar_box(self.lidar_box, self.db)


if __name__ == "__main__":
    unittest.main()
