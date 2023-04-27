from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_is_comfortable import EgoIsComfortableStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_jerk import EgoJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_acceleration import EgoLatAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration import EgoLonAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_jerk import EgoLonJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_acceleration import EgoYawAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_rate import EgoYawRateStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_is_comfortable/ego_is_comfortable.json')
def test_ego_is_comfortable(scene: Dict[str, Any]) -> None:
    """
    Tests ego is comfortable by checking if it is the expected comfortable.
    :param scene: the json scene
    """
    ego_jerk_metric = EgoJerkStatistics('ego_jerk', 'Dynamics', max_abs_mag_jerk=8.37)
    ego_lat_accel_metric = EgoLatAccelerationStatistics('ego_lat_accel', 'Dynamics', max_abs_lat_accel=4.89)
    ego_lon_accel_metric = EgoLonAccelerationStatistics(
        'ego_lon_accel', 'Dynamics', min_lon_accel=-4.05, max_lon_accel=2.40
    )
    ego_lon_jerk_metric = EgoLonJerkStatistics('ego_lon_jerk', 'dynamic', max_abs_lon_jerk=4.13)
    ego_yaw_accel_metric = EgoYawAccelerationStatistics('ego_yaw_accel', 'dynamic', max_abs_yaw_accel=1.93)
    ego_yaw_rate_metric = EgoYawRateStatistics('ego_yaw_rate', 'dynamic', max_abs_yaw_rate=0.95)
    metric = EgoIsComfortableStatistics(
        name='ego_is_comfortable_statistics',
        category='Dynamics',
        ego_jerk_metric=ego_jerk_metric,
        ego_lat_acceleration_metric=ego_lat_accel_metric,
        ego_lon_acceleration_metric=ego_lon_accel_metric,
        ego_lon_jerk_metric=ego_lon_jerk_metric,
        ego_yaw_acceleration_metric=ego_yaw_accel_metric,
        ego_yaw_rate_metric=ego_yaw_rate_metric,
    )

    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
