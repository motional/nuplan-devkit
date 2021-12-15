import os

import hydra
import pytest
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.utils.testing_utils import setup_history

CONFIG_PATH = os.path.join('..', '..', '..', '..', 'script/config/common/simulation_metric/common/')
CONFIG_NAME = 'time_to_collision_statistics'


@nuplan_test(path='json/time_to_collision/time_to_collision.json')
def test_time_to_collision(scene) -> None:  # type: ignore
    """
    Test predicted time to collision

    :param scene: the json scene
    """

    history = setup_history(scene)

    # Metric
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)
    cfg = hydra.compose(config_name=CONFIG_NAME)
    time_to_collision = hydra.utils.instantiate(cfg)['time_to_collision_statistics']

    result = time_to_collision.compute(history)[0]
    expected_times_to_collision = [float(t) for t in scene['expected']["times_to_collision"]]

    for i, (actual_ttc, expected_ttc) in enumerate(zip(result.time_series.values, expected_times_to_collision)):
        assert round(actual_ttc, 2) == expected_ttc, f"Wrong TTC for timestep {i}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
