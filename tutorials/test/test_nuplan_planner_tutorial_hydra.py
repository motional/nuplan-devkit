import signal
import tempfile
import unittest
from os import path
from pathlib import Path

import hydra
import ray
from tutorials.utils.tutorial_utils import construct_nuboard_hydra_paths, construct_simulation_hydra_paths

from nuplan.planning.script.run_nuboard import main as main_nuboard
from nuplan.planning.script.run_simulation import main as main_simulation

TEST_TIMEOUT = 10  # [s] timeout dashboard after this duration

# relative path to tutorial to pass configs to hydra in notebook
TUTORIAL_PATH_REL = path.dirname(path.dirname(path.relpath(__file__)))
# to allow bazel to find tutorial file for testing
TUTORIAL_PATH_ABS = path.dirname(path.dirname(path.realpath(__file__)))
# path to hydra configs
BASE_CONFIG_PATH = path.join(TUTORIAL_PATH_ABS, '../nuplan/planning/script')


def _timeout_handler(signum: int, frame: 'frame') -> None:  # type: ignore # noqa
    """
    Signal handler for timing out execution through exception.
    """
    raise TimeoutError


class TestPlannerTutorialHydra(unittest.TestCase):
    """
    Test planner tutorial Jupyter notebook hydra configuration.
    """

    def setUp(self) -> None:
        """Setup."""
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Clean up."""
        if Path(self.tmp_dir.name).exists():
            self.tmp_dir.cleanup()

        # Stop ray
        if ray.is_initialized():
            ray.shutdown()

    def test_hydra_paths_utils(self) -> None:
        """
        Test HydraConfigPaths utility functions for storing config paths for simulation and visualization.
        """
        # test simulation
        simulation_hydra_paths = construct_simulation_hydra_paths(BASE_CONFIG_PATH)

        with hydra.initialize_config_dir(config_dir=simulation_hydra_paths.config_path):
            cfg = hydra.compose(
                config_name=simulation_hydra_paths.config_name,
                overrides=[
                    f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
                    '+simulation=open_loop_boxes',
                    'log_config=false',
                    'scenario_builder=nuplan_mini',
                    'planner=simple_planner',
                    'scenario_filter=one_of_each_scenario_type',
                    'scenario_filter.limit_total_scenarios=2',
                    'exit_on_failure=true',
                    """selected_simulation_metrics='[ego_acceleration_statistics, ego_jerk_statistics]'""",
                    f'group={self.tmp_dir.name}',
                    'experiment_name=hydra_paths_utils_test',
                    'output_dir=${group}/${experiment}',
                ],
            )
            main_simulation(cfg)

        # test nuboard
        results_dir = Path(cfg.output_dir)
        simulation_file = [str(file) for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard'][
            0
        ]

        nuboard_hydra_paths = construct_nuboard_hydra_paths(BASE_CONFIG_PATH)

        # Create timeout alarm signal to preempt the dashboard's IO loop and test for initialization errors
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TEST_TIMEOUT)

        try:
            with hydra.initialize_config_dir(config_dir=nuboard_hydra_paths.config_path):
                cfg = hydra.compose(
                    config_name=nuboard_hydra_paths.config_name,
                    overrides=[
                        'scenario_builder=nuplan_mini',
                        f'simulation_path={simulation_file}',
                        f'hydra.searchpath=[{nuboard_hydra_paths.common_dir}, {nuboard_hydra_paths.experiment_dir}]',
                        'port_number=4555',
                    ],
                )
                main_nuboard(cfg)

        except Exception as exc:  # noqa
            signal.alarm(0)  # Stop alarm, if the exception did not come due to timeout
            self.assertTrue(isinstance(exc, TimeoutError))  # Exception is due to timeout and not due to other error


if __name__ == '__main__':
    unittest.main()
