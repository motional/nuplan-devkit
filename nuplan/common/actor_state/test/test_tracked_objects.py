import unittest
from typing import Any, Dict

from nuplan.common.actor_state.test.test_utils import get_sample_agent
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType


class TestTrackedObjects(unittest.TestCase):
    """Tests TrackedObjects class"""

    def setUp(self) -> None:
        """Creates sample agents for testing"""
        self.agents = [
            get_sample_agent('foo', TrackedObjectType.PEDESTRIAN),
            get_sample_agent('bar', TrackedObjectType.VEHICLE),
            get_sample_agent('bar_out_the_car', TrackedObjectType.PEDESTRIAN),
        ]

    def test_construction(self) -> None:
        """Tests that the object can be created correctly."""
        tracked_objects = TrackedObjects(self.agents)

        expected_type_and_set_of_tokens: Dict[TrackedObjectType, Any] = {
            object_type: set() for object_type in TrackedObjectType
        }
        expected_type_and_set_of_tokens[TrackedObjectType.PEDESTRIAN].update({'foo', 'bar_out_the_car'})
        expected_type_and_set_of_tokens[TrackedObjectType.VEHICLE].update({'bar'})

        for tracked_object_type in TrackedObjectType:
            if tracked_object_type not in expected_type_and_set_of_tokens:
                continue

            self.assertEqual(
                expected_type_and_set_of_tokens[tracked_object_type],
                {
                    tracked_object.token
                    for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type)
                },
            )

    def test_get_subset(self) -> None:
        """Tests that the object can be created correctly."""
        tracked_objects = TrackedObjects(self.agents)

        agents = tracked_objects.get_agents()
        static_objects = tracked_objects.get_static_objects()

        self.assertEqual(3, len(agents))
        self.assertEqual(0, len(static_objects))

    def test_get_tracked_objects_of_types(self) -> None:
        """Test get_tracked_objects_of_types()"""
        tracked_objects = TrackedObjects(self.agents)
        track_types = [TrackedObjectType.PEDESTRIAN, TrackedObjectType.VEHICLE]
        tracks = tracked_objects.get_tracked_objects_of_types(track_types)

        self.assertEqual(3, len(tracks))


if __name__ == '__main__':
    unittest.main()
