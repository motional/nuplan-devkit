from typing import Any, Dict

import pytest

from nuplan.common.geometry.interpolate_tracked_object import interpolate_tracks
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.utils.serialization.from_scene import from_scene_to_tracked_objects_with_predictions


@nuplan_test(path='json/interpolate_future.json')
def test_interpolate_tracked_object(scene: Dict[str, Any]) -> None:
    """Test that we can interpolate agents with various initial length."""
    tracked_objects = from_scene_to_tracked_objects_with_predictions(scene["world"], scene["prediction"])

    # Interpolate agents
    future_horizon_len_s = 8.0
    future_interval_s = 0.5
    agents = interpolate_tracks(tracked_objects, future_horizon_len_s, future_interval_s)

    # Compute the desired length of predictions, we include also initial state into waypoints
    desired_length = int(future_horizon_len_s / future_interval_s) + 1
    for agent in agents:
        assert agent.predictions, "Predictions have to exist!"
        for prediction in agent.predictions:
            # Extract the GT for last prediction time stamp
            last_original_prediction_state = [
                json_prediction["states"][-1]
                for json_prediction in scene["prediction"]
                if json_prediction["id"] == agent.metadata.track_id
            ]
            assert len(last_original_prediction_state) == 1, "We did not find original prediction?"
            last_time_stamp = last_original_prediction_state[0]["timestamp"]
            # Make sure that if we have short predictions, only one state will be present
            nonzero = [w for w in prediction.waypoints if w]
            if last_time_stamp < future_interval_s:
                assert len(nonzero) == 1, "The length of non zero predictions has to be 1!"

            # If the length spans at least beyond one interval, we expect 2 states, the scene was designed to have one
            if future_interval_s < last_time_stamp < 2 * future_interval_s:
                assert len(nonzero) == 2, "The length of non zero predictions has to be 2!"

            # Check the desired length and the sampling
            assert len(prediction.waypoints) == desired_length, "Prediction does not have desired length!"
            for index, waypoint in enumerate(prediction.waypoints):
                if index != 0 and waypoint:
                    time_interval = waypoint.time_point.time_s - prediction.waypoints[index - 1].time_point.time_s
                    assert time_interval == pytest.approx(future_interval_s), "The sampling is not correct!"


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
