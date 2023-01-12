import unittest
from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nuplan.planning.script.run_metric import CONFIG_NAME
from nuplan.planning.script.run_metric import main as run_metric
from nuplan.planning.script.run_metric_aggregator import CONFIG_NAME as METRIC_AGGREGATOR_CONFIG_NAME
from nuplan.planning.script.run_metric_aggregator import main as run_metric_aggregator
from nuplan.planning.script.run_simulation import main as run_simulation
from nuplan.planning.script.test.skeleton_test_simulation import SkeletonTestSimulation


class TestRunMetric(SkeletonTestSimulation):
    """Test running metrics only."""

    def test_run_simulation_fails_with_no_logs(self) -> None:
        """Sanity test to test that metric_runner fails to run when there is no simulation logs."""
        # Run simulations
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    '+simulation=open_loop_boxes',
                    f'simulation_log_main_path={self.tmp_dir.name}',
                    'experiment_name=simulation_no_metric_test',
                ],
            )
            with self.assertRaises(FileNotFoundError):
                run_metric(cfg)

    def test_run_simulation_logs(self) -> None:
        """Sanity test to run simulation logs by computing metrics only."""
        # Run simulation
        with initialize_config_dir(config_dir=self.config_path):

            # Run without metrics
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    '+simulation=open_loop_boxes',
                    'run_metric=false',
                    'experiment_name=open_loop_boxes/simulation_metric_test',
                    'worker=sequential',
                    'main_callback=[time_callback]',
                ],
            )
            run_simulation(cfg)

            # Reuse the previous exp_output_dir and change the simulation_log_main_path to run metrics only
            exp_output_dir = deepcopy(cfg.output_dir)
            OmegaConf.set_struct(cfg, False)
            cfg.simulation_log_main_path = exp_output_dir
            OmegaConf.set_struct(cfg, True)
            run_metric(cfg)

        # Run metric aggregator
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=METRIC_AGGREGATOR_CONFIG_NAME,
                overrides=[f'output_dir={exp_output_dir}', "challenges=['open_loop_boxes']"],
            )
            run_metric_aggregator(cfg)
            # Check file output
            metric_aggregator_output = Path(cfg.aggregator_save_path)
            aggregator_output_file_length = len(list(metric_aggregator_output.rglob("*")))
            # Only one for the open_loop_boxes challenge
            self.assertEqual(aggregator_output_file_length, 1)


if __name__ == '__main__':
    unittest.main()
