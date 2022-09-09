import os
import signal
import tempfile
import unittest
from pathlib import Path

import ray
from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_nuboard import CONFIG_NAME as NUBOARD_CONFIG_NAME
from nuplan.planning.script.run_nuboard import main as nuboard_main
from nuplan.planning.script.run_simulation import CONFIG_NAME as SIMULATION_CONFIG_NAME
from nuplan.planning.script.run_simulation import main as simulation_main

TEST_TIMEOUT = 10  # [s] timeout dashboard after this duration


def _timeout_handler(signum: int, frame: 'frame') -> None:  # type: ignore # noqa
    """
    Signal handler for timing out execution through exception.
    """
    raise TimeoutError


class TestRunNuBoard(unittest.TestCase):
    """
    Test running main nuboard entry point.
    """

    def setUp(self) -> None:
        """Set up basic config."""
        main_path = os.path.dirname(os.path.realpath(__file__))
        self.simulation_config_path = os.path.join(main_path, '../config/simulation/')
        self.nuboard_config_path = os.path.join(main_path, '../config/nuboard/')

        # TODO: Investigate pkg in hydra
        # Since we are not using the default config in this test, we need to specify the Hydra search path in the
        # compose API override, otherwise the Jenkins build fails because bazel cannot find the simulation config file.
        common_dir = 'file://' + os.path.join(main_path, '..', 'config', 'common')
        experiment_dir = 'file://' + os.path.join(main_path, '..', 'experiments')
        self.search_path = f'hydra.searchpath=[{common_dir}, {experiment_dir}]'

        self.tmp_dir = tempfile.TemporaryDirectory()
        if not os.getenv("NUPLAN_EXP_ROOT", None):
            os.environ["NUPLAN_EXP_ROOT"] = self.tmp_dir.name

        # Default hydra overrides for quick unit testing
        self.simulation_overrides = [
            'log_config=false',
            'scenario_builder=nuplan_mini',
            'planner=simple_planner',
            'scenario_filter=one_of_each_scenario_type',
            'scenario_filter.limit_total_scenarios=2',
            'exit_on_failure=true',
            """selected_simulation_metrics='[ego_acceleration_statistics, ego_jerk_statistics]'""",
            f'group={self.tmp_dir.name}',
            'output_dir=${group}/${experiment}',
        ]

    def tearDown(self) -> None:
        """Clean up."""
        if Path(self.tmp_dir.name).exists():
            self.tmp_dir.cleanup()

        # Stop ray
        if ray.is_initialized():
            ray.shutdown()

    def test_nuboard_incorrect_file(self) -> None:
        """
        Tests that the nuboard correctly recognizes incorrect file extensions.
        """
        with self.assertRaises(RuntimeError):
            with initialize_config_dir(config_dir=self.nuboard_config_path):
                cfg = compose(config_name=NUBOARD_CONFIG_NAME, overrides=[self.search_path, 'simulation_path=test.tmp'])
                nuboard_main(cfg)

    def test_nuboard_integration(self) -> None:
        """
        Sanity test for launching the nuboard using simulation results file.
        """
        with initialize_config_dir(config_dir=self.simulation_config_path):
            cfg = compose(
                config_name=SIMULATION_CONFIG_NAME,
                overrides=[self.search_path, *self.simulation_overrides, '+simulation=open_loop_boxes'],
            )
            simulation_main(cfg)

        results_dir = list(list(Path(self.tmp_dir.name).iterdir())[0].iterdir())[0]  # get the child dir 2 levels in
        simulation_dir = results_dir / 'simulation_log'
        planner_dir = list(simulation_dir.iterdir())[0]
        scenario_dir = list(planner_dir.iterdir())[0]
        log_dir = list(scenario_dir.iterdir())[0]
        scene_dir = list(log_dir.iterdir())[0]
        scene_file = list(scene_dir.iterdir())[0]
        nuboard_file = [file for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]
        self.assertTrue(scene_file.is_file())
        self.assertEqual(scene_file.suffix, '.xz')

        # Create timeout alarm signal to preempt the dashboard's IO loop and test for initialization errors
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TEST_TIMEOUT)

        try:
            with initialize_config_dir(config_dir=self.nuboard_config_path):
                cfg = compose(
                    config_name=NUBOARD_CONFIG_NAME,
                    overrides=[self.search_path, f'simulation_path={str(nuboard_file)}', 'port_number=4554'],
                )
                nuboard_main(cfg)
        except Exception as exc:  # noqa
            signal.alarm(0)  # Stop alarm, if the exception did not come due to timeout
            self.assertTrue(isinstance(exc, TimeoutError))  # Exception is due to timeout and not due to other error


if __name__ == '__main__':
    unittest.main()
