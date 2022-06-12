import unittest
from unittest.mock import Mock, patch

from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata


class TestSceneObject(unittest.TestCase):
    """Tests SceneObject class"""

    @patch("nuplan.common.actor_state.tracked_objects_types.TrackedObjectType")
    @patch("nuplan.common.actor_state.oriented_box.OrientedBox")
    def test_initialization(self, mock_box: Mock, mock_tracked_object_type: Mock) -> None:
        """Tests that agents can be initialized correctly"""
        scene_object = SceneObject(mock_tracked_object_type, mock_box, SceneObjectMetadata(1, "123", 1, "456"))
        self.assertEqual("123", scene_object.token)
        self.assertEqual("456", scene_object.track_token)
        self.assertEqual(mock_box, scene_object.box)
        self.assertEqual(mock_tracked_object_type, scene_object.tracked_object_type)

    @patch("nuplan.common.actor_state.scene_object.StateSE2")
    @patch("nuplan.common.actor_state.scene_object.OrientedBox")
    @patch("nuplan.common.actor_state.scene_object.TrackedObjectType")
    @patch("nuplan.common.actor_state.scene_object.SceneObject.__init__")
    def test_construction(self, mock_init: Mock, mock_type: Mock, mock_box_object: Mock, mock_state: Mock) -> None:
        """Test that agents can be constructed correctly."""
        mock_init.return_value = None
        mock_box = Mock()
        mock_box_object.return_value = mock_box
        _ = SceneObject.from_raw_params("123", "123", 1, 1, mock_state, size=(3, 2, 1))

        mock_box_object.assert_called_with(mock_state, width=3, length=2, height=1)
        mock_init.assert_called_with(
            metadata=SceneObjectMetadata(token="123", track_token="123", timestamp_us=1, track_id=1),
            tracked_object_type=mock_type.GENERIC_OBJECT,
            oriented_box=mock_box,
        )


if __name__ == '__main__':
    unittest.main()
