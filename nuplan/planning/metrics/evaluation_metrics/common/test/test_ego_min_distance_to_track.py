from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_min_distance_to_track import EgoMinDistanceToTrackStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_min_distance_to_track/ego_min_distance_to_track.json')
def test_ego_min_distance_to_track(scene: Dict[str, Any]) -> None:
    """
    Tests ego min distance to track statistics as expected.
    :param scene: the json scene
    """
    metric = EgoMinDistanceToTrackStatistics('ego_min_distance_track_statistics', 'Dynamics')
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
