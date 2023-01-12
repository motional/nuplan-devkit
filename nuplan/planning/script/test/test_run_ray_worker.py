import unittest

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_simulation import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_simulation import SkeletonTestSimulation


class TestRunRayWorker(SkeletonTestSimulation):
    """Test running ray workers in simulation."""

    def test_ray_worker(self) -> None:
        """
        Sanity test for ray worker.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'worker=ray_distributed',
                    'worker.debug_mode=true',
                    'scenario_filter.limit_total_scenarios=2',
                    """selected_simulation_metrics='[ego_acceleration_statistics, ego_jerk_statistics]'""",
                    '+simulation=open_loop_boxes',
                ],
            )
            main(cfg)


if __name__ == '__main__':
    unittest.main()
