import unittest
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
from matplotlib.axes import Axes

from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.tests.test_utils_nuplan_db import get_test_nuplan_db, get_test_nuplan_image
from nuplan.database.utils.boxes.box3d import Box3D


class TestImage(unittest.TestCase):
    """Test class Image"""

    def setUp(self) -> None:
        """
        Initializes a test Image
        """
        self.db = get_test_nuplan_db()
        self.image = get_test_nuplan_image()

    @patch("nuplan.database.nuplan_db_orm.image.inspect", autospec=True)
    def test_session(self, inspect: Mock) -> None:
        """
        Tests _session method
        """
        # Setup
        mock_session = PropertyMock()
        inspect.return_value = Mock()
        inspect.return_value.session = mock_session

        # Call method under test
        result = self.image._session()

        # Assertions
        inspect.assert_called_once_with(self.image)
        mock_session.assert_called_once()
        self.assertEqual(result, mock_session.return_value)

    @patch("nuplan.database.nuplan_db_orm.image.simple_repr", autospec=True)
    def test_repr(self, simple_repr: Mock) -> None:
        """
        Tests string representation
        """
        # Call method under test
        result = self.image.__repr__()

        # Assertions
        simple_repr.assert_called_once_with(self.image)
        self.assertEqual(result, simple_repr.return_value)

    def test_log(self) -> None:
        """
        Tests property log
        """
        # Call method under test
        log = self.image.log

        # Assertions
        self.assertEqual(log, self.image.camera.log)

    @patch("nuplan.database.nuplan_db_orm.image.func")
    @patch("nuplan.database.nuplan_db_orm.image.Image._session")
    def test_lidar_pc(self, mock_session: Mock, mock_func: Mock) -> None:
        """
        Tests property lidar_pc
        """
        # Set up
        mock_query = mock_session.query
        mock_lidar_pc = mock_query.return_value
        mock_lidar_pc_ordered = mock_lidar_pc.order_by.return_value
        mock_first = mock_lidar_pc_ordered.first.return_value

        # Call method under test
        result = self.image.lidar_pc

        # Assertions
        mock_query.assert_called_once_with(LidarPc)
        self.assertTrue(mock_func.abs.call_args[0][0].compare(LidarPc.timestamp - self.image.timestamp))
        mock_lidar_pc.order_by.assert_called_once_with(mock_func.abs.return_value)
        mock_lidar_pc_ordered.first.assert_called_once()
        self.assertEqual(result, mock_first)

    @patch("nuplan.database.nuplan_db_orm.image.Image.lidar_pc")
    def test_lidar_boxes(self, mock_lidar_pc: Mock) -> None:
        """
        Tests property lidar_boxes
        """
        # Call method under test
        result = self.image.lidar_boxes

        # Assertions
        self.assertEqual(result, mock_lidar_pc.lidar_boxes)

    @patch("nuplan.database.nuplan_db_orm.image.Image.lidar_pc")
    def test_scene(self, mock_lidar_pc: Mock) -> None:
        """
        Tests property scene
        """
        # Call method under test
        result = self.image.scene

        # Assertions
        self.assertEqual(result, mock_lidar_pc.scene)

    @patch("nuplan.database.nuplan_db_orm.image.PIL.Image.open")
    @patch("nuplan.database.nuplan_db_orm.image.Image.load_bytes_jpg")
    def test_load_as_pil(self, mock_load_bytes: Mock, mock_pil_open: Mock) -> None:
        """
        Tests load_as with PIL image type
        """
        # Setup
        mock_db = Mock()

        # Call method under test
        img = self.image.load_as(mock_db, "pil")

        # Assertions
        mock_load_bytes.assert_called_once_with(mock_db)
        mock_pil_open.assert_called_once_with(mock_load_bytes.return_value)
        self.assertEqual(img, mock_pil_open.return_value)

    @patch("nuplan.database.nuplan_db_orm.image.np.array")
    @patch("nuplan.database.nuplan_db_orm.image.PIL.Image.open")
    @patch("nuplan.database.nuplan_db_orm.image.Image.load_bytes_jpg")
    def test_load_as_np(self, mock_load_bytes: Mock, mock_pil_open: Mock, mock_np_array: Mock) -> None:
        """
        Tests load_as with numpy array image type
        """
        # Setup
        mock_db = Mock()

        # Call method under test
        img = self.image.load_as(mock_db, "np")

        # Assertions
        mock_load_bytes.assert_called_once_with(mock_db)
        mock_pil_open.assert_called_once_with(mock_load_bytes.return_value)
        mock_np_array.assert_called_once_with(mock_pil_open.return_value)
        self.assertEqual(img, mock_np_array.return_value)

    @patch("nuplan.database.nuplan_db_orm.image.cv2.COLOR_RGB2BGR")
    @patch("nuplan.database.nuplan_db_orm.image.cv2.cvtColor")
    @patch("nuplan.database.nuplan_db_orm.image.np.array")
    @patch("nuplan.database.nuplan_db_orm.image.PIL.Image.open")
    @patch("nuplan.database.nuplan_db_orm.image.Image.load_bytes_jpg")
    def test_load_as_cv2(
        self, mock_load_bytes: Mock, mock_pil_open: Mock, mock_np_array: Mock, mock_cvtColor: Mock, mock_rgb2bgr: Mock
    ) -> None:
        """
        Tests load_as with cv2 image type
        """
        # Setup
        mock_db = Mock()

        # Call method under test
        img = self.image.load_as(mock_db, "cv2")

        # Assertions
        mock_load_bytes.assert_called_once_with(mock_db)
        mock_pil_open.assert_called_once_with(mock_load_bytes.return_value)
        mock_np_array.assert_called_once_with(mock_pil_open.return_value)
        mock_cvtColor.assert_called_once_with(mock_np_array.return_value, mock_rgb2bgr)
        self.assertEqual(img, mock_cvtColor.return_value)

    def test_load_as_invalid(self) -> None:
        """
        Tests load_as with invalid image type
        """
        # Setup
        mock_db = Mock()

        # Call method under test
        with self.assertRaises(AssertionError):
            self.image.load_as(mock_db, "invalid")

    def test_filename(self) -> None:
        """
        Tests property filename
        """
        # Call method under test
        filename = self.image.filename

        # Assertions
        self.assertEqual(filename, self.image.filename_jpg)

    @patch("nuplan.database.nuplan_db_orm.image.osp.join")
    @patch("nuplan.database.nuplan_db_orm.image.Image.filename")
    def test_load_bytes_jpg(self, mock_filename: Mock, mock_osp_join: Mock) -> None:
        """
        Tests method to load bytes of the jpg data db.load_blob(osp.join("sensor_blobs", self.filename))
        """
        # Setup
        mock_load_blob = Mock()
        mock_db = Mock(load_blob=mock_load_blob)

        # Call the method under test
        result = self.image.load_bytes_jpg(mock_db)

        # Assertions
        mock_osp_join.assert_called_once_with("sensor_blobs", mock_filename)
        mock_load_blob.assert_called_once_with(mock_osp_join.return_value)
        self.assertEqual(result, mock_load_blob.return_value)

    @patch("nuplan.database.nuplan_db_orm.image.osp.join")
    def test_path(self, mock_osp_join: Mock) -> None:
        """
        Tests image path based on DB data root
        """
        # Setup
        mock_db = Mock(data_root="data_root")

        # Call method under test
        path = self.image.path(mock_db)

        # Assertions
        mock_osp_join.assert_called_once_with("data_root", self.image.filename)
        self.assertEqual(path, mock_osp_join.return_value)

    @patch("nuplan.database.nuplan_db_orm.image.get_boxes")
    @patch("nuplan.database.nuplan_db_orm.image.Image.camera")
    @patch("nuplan.database.nuplan_db_orm.image.Image.ego_pose")
    def test_boxes(self, mock_egopose: Mock, mock_camera: Mock, mock_get_boxes: Mock) -> None:
        """
        Test loading of boxes associated with this Image
        """
        # Call method under test
        boxes = self.image.boxes("Frame")

        # Assertions
        mock_get_boxes.assert_called_once_with(
            self.image, "Frame", mock_egopose.trans_matrix_inv, mock_camera.trans_matrix_inv
        )
        self.assertEqual(boxes, mock_get_boxes.return_value)

    def test_future_ego_poses(self) -> None:
        """
        Test method to get n future poses
        """
        # Setup
        n_ego_poses = 4

        # Call method under test
        future_ego_poses = self.image.future_or_past_ego_poses(number=n_ego_poses, mode="n_poses", direction="next")

        # Assertions
        ego_pose = self.image.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(
                ego_pose.timestamp,
                future_ego_poses[i].timestamp,
                "Timestamps of current EgoPose must be less that future EgoPoses.",
            )

    def test_past_ego_poses(self) -> None:
        """
        Test method to get n past poses
        """
        # Setup
        n_ego_poses = 4

        # Call method under test
        past_ego_poses = self.image.future_or_past_ego_poses(number=n_ego_poses, mode="n_poses", direction="prev")

        # Assertions
        ego_pose = self.image.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(
                ego_pose.timestamp,
                past_ego_poses[i].timestamp,
                "Timestamps of current EgoPose must be greater than past EgoPoses ",
            )

    def test_invalid_ego_poses(self) -> None:
        """
        Test method to get poses, with invalid inputs
        """
        with self.assertRaises(ValueError):
            self.image.future_or_past_ego_poses(number=1, mode="n_poses", direction="invalid")

        with self.assertRaises(NotImplementedError):
            self.image.future_or_past_ego_poses(number=1, mode="invalid", direction="prev")

    @patch("nuplan.database.nuplan_db_orm.image.Image.boxes")
    @patch("nuplan.database.nuplan_db_orm.image.box_in_image")
    @patch("nuplan.database.nuplan_db_orm.image.Image.load_as", autospec=True)
    def test_render(self, mock_load: Mock, mock_box_in_image: Mock, mock_boxes: Mock) -> None:
        """
        Test render method
        """
        # Setup
        mock_ax = Mock(spec=Axes)

        mock_box = Mock(spec=Box3D, token="token")
        mock_box.render = Mock()
        mock_boxes.return_value = [mock_box]

        mock_db = MagicMock()
        mock_db.lidar_box = MagicMock()

        mock_box_in_image.return_value = True

        # Call method under test
        self.image.render(mock_db, with_3d_anns=True, box_vis_level="box_vis_level", ax=mock_ax)

        # Assertions
        mock_boxes.assert_called_once()

        self.assertEqual(mock_box_in_image.call_args.args[0], mock_box)
        np.testing.assert_array_equal(mock_box_in_image.call_args.args[1], self.image.camera.intrinsic_np)
        np.testing.assert_array_equal(
            mock_box_in_image.call_args.args[2], (self.image.camera.width, self.image.camera.height)
        )
        self.assertEqual(mock_box_in_image.call_args.kwargs, {"vis_level": "box_vis_level"})

        mock_box.render.assert_called_once()


if __name__ == "__main__":
    unittest.main()
