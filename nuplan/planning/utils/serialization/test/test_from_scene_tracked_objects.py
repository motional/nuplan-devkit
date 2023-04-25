import sys
from typing import Any, Dict

import pytest

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.utils.serialization.from_scene import from_scene_to_tracked_objects_with_scene_predictions


@nuplan_test(path='json/load_from_scene.json')
def test_load_from_scene(scene: Dict[str, Any]) -> None:
    """
    Tests loading tracked objects with predictions from a scene json.
    :param scene: The input scene loaded from the json file.
    """
    tracked_objects = from_scene_to_tracked_objects_with_scene_predictions(scene)

    agent = tracked_objects.tracked_objects[0]

    assert agent.track_token == "0"
    assert agent.tracked_object_type == TrackedObjectType.VEHICLE
    assert list(agent.box.center) == [1, 2, 0]
    assert len(agent.predictions) == 2
    assert agent.predictions[0].probability == 0.9
    assert agent.predictions[1].probability == 0.1
    assert agent.box.width == 2.0
    assert agent.box.length == 4.7

    # Check full prediction for the first case
    for i, state in enumerate(agent.predictions[0].waypoints):
        assert list(state.center) == pytest.approx([1 + 0.01 * i, 2 + 0.01 * i, 0.01 * i])
        assert state.time_us == pytest.approx(agent.metadata.timestamp_us + int(0.5 * i * 1e6))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__] + sys.argv[1:], plugins=[NUPLAN_TEST_PLUGIN]))
