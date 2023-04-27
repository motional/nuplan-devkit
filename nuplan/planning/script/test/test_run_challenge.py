import unittest

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_simulation import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_simulation import SkeletonTestSimulation


class TestRunChallenge(SkeletonTestSimulation):
    """Test main simulation entry point across different challenges."""

    def test_simulation_challenge_1(self) -> None:
        """
        Sanity check for challenge 1 simulation.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'worker=single_machine_thread_pool',
                    'worker.use_process_pool=true',
                    '+simulation=open_loop_boxes',
                ],
            )
            main(cfg)


if __name__ == '__main__':
    unittest.main()
