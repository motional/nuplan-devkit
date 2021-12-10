import os
import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from nuplan.planning.script.builders.planner_builder import build_planner
from nuplan.planning.script.run_simulation import CONFIG_NAME, main, run_simulation
from omegaconf import OmegaConf


class TestSimulation(unittest.TestCase):
    """
    Test main simulation entry point across different challenges.
    """

    def setUp(self) -> None:
        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/simulation/')

        # Since we are not using the default config in this test, we need to specify the Hydra search path in the
        # compose API override, otherwise the Jenkins build fails because bazel cannot find the simulation config file.
        common_dir = "file://" + os.path.join(main_path, '..', 'config', 'common')
        experiment_dir = "file://" + os.path.join(main_path, '..', 'experiments')
        self.search_path = f'hydra.searchpath=[{common_dir}, {experiment_dir}]'
        self.tmp_dir = tempfile.TemporaryDirectory()
        # Default hydra overrides for quick unit testing
        self.default_overrides = [
            'log_config=false',
            'scenario_builder=nuplan_mini',
            'planner=simple_planner',
            'scenario_builder/nuplan/scenario_filter=one_hand_picked_scenario',
            'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.05',
            'exit_on_failure=true',
            f"group={self.tmp_dir.name}"
        ]

    def tearDown(self) -> None:
        if Path(self.tmp_dir.name).exists():
            self.tmp_dir.cleanup()

    def test_simulation_challenge_1(self) -> None:
        """
        Sanity test for challenge 1 simulation.
        """

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME,
                          overrides=[self.search_path,
                                     *self.default_overrides,
                                     'worker.local_mode=true',
                                     '+simulation=challenge_1_open_loop_boxes'])
            main(cfg)

    def test_worker_sequential(self) -> None:
        """
        Sanity test for sequential worker.
        """

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME,
                          overrides=[
                              self.search_path,
                              *self.default_overrides,
                              'worker=sequential',
                              """selected_simulation_metrics='[ego_acceleration_statistics, ego_jerk_statistics]'""",
                              '+simulation=challenge_1_open_loop_boxes'])
            main(cfg)

    def test_worker_parallel(self) -> None:
        """
        Sanity test parallel worker.
        """

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME,
                          overrides=[self.search_path,
                                     *self.default_overrides,
                                     'worker=single_machine_thread_pool',
                                     '+simulation=challenge_1_open_loop_boxes'])
            main(cfg)

    def test_ray_worker(self) -> None:
        """
        Sanity test for ray worker.
        """

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME,
                          overrides=[self.search_path,
                                     *self.default_overrides,
                                     'worker=ray_distributed',
                                     'worker.local_mode=true',
                                     '+simulation=challenge_1_open_loop_boxes'])
            main(cfg)

    def test_run_simulation(self) -> None:
        """
        Sanity test for passing planner as argument to run_simulation
        """

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME,
                          overrides=[self.search_path,
                                     *self.default_overrides,
                                     'observation=box_observation',
                                     'ego_controller=log_play_back_controller',
                                     'experiment_name=simulation_test'])
            planner_cfg = cfg.planner
            planner = build_planner(planner_cfg)
            OmegaConf.set_struct(cfg, False)
            cfg.pop('planner')
            OmegaConf.set_struct(cfg, True)
            run_simulation(cfg, planner)


if __name__ == '__main__':
    unittest.main()
