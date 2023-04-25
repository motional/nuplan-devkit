from typing import Any, Dict

import numpy as np
import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
    CollisionType,
    EgoAtFaultCollisionStatistics,
)
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/no_ego_at_fault_collision/no_collision.json')
def test_no_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is no collision as expected.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_result = metric_statistic_test(scene=scene, metric=metric)
    statistics = metric_result.statistics

    # No at_fault collisions
    assert statistics[1].value == 0
    # No collisions
    assert len(metric.all_collisions) == 0


@nuplan_test(path='json/no_ego_at_fault_collision/active_front_collision.json')
def test_active_front_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one front collision in this scene.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)

    # 1 active_front_collision
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert (
        list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type
        == CollisionType.ACTIVE_FRONT_COLLISION
    )


@nuplan_test(path='json/no_ego_at_fault_collision/active_lateral_collision.json')
def test_active_lateral_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one lateral collision in this scene which is at fault.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)

    # 1 active_lateral_collision when ego is making a lane change
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert (
        list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type
        == CollisionType.ACTIVE_LATERAL_COLLISION
    )


@nuplan_test(path='json/no_ego_at_fault_collision/active_rear_collision.json')
def test_active_rear_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one rear collision in this scene which is not at fault.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)

    # 1 active_rear_collision when ego is not making a lane change
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert (
        list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type
        == CollisionType.ACTIVE_REAR_COLLISION
    )


@nuplan_test(path='json/no_ego_at_fault_collision/stopped_track_collision.json')
def test_stopped_track_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one collision with a stopped track in this scene which is at fault.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)

    # 1 collision_at_stopped_track
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert (
        list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type
        == CollisionType.STOPPED_TRACK_COLLISION
    )


@nuplan_test(path='json/no_ego_at_fault_collision/stopped_ego_collision.json')
def test_stopped_ego_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one collision when ego is stopped in this scene which is not at fault.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)

    # 1 collision_at_stopped_track
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert (
        list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type
        == CollisionType.STOPPED_EGO_COLLISION
    )


@nuplan_test(path='json/no_ego_at_fault_collision/multiple_collisions.json')
def test_multiple_collisions(scene: Dict[str, Any]) -> None:
    """
    Tests there are 4 tracks and 3 collisions in this scene, and there are 2 at-fault-collisions for which
    we find the violation metric.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)

    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 3
    # 1 active_lateral_collision when ego is not making a lane change at the second timestamp
    assert (
        list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type
        == CollisionType.ACTIVE_LATERAL_COLLISION
    )
    # 2 active_front_collisions at the last timestamp
    assert (
        list(metric.all_collisions[1].collisions_id_data.values())[0].collision_type
        == CollisionType.ACTIVE_FRONT_COLLISION
    )
    assert (
        list(metric.all_collisions[1].collisions_id_data.values())[1].collision_type
        == CollisionType.ACTIVE_FRONT_COLLISION
    )


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
