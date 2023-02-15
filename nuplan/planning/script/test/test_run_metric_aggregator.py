import unittest
from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_metric import CONFIG_NAME
from nuplan.planning.script.run_metric_aggregator import CONFIG_NAME as METRIC_AGGREGATOR_CONFIG_NAME
from nuplan.planning.script.run_metric_aggregator import main as run_metric_aggregator
from nuplan.planning.script.run_simulation import main as run_simulation
from nuplan.planning.script.test.skeleton_test_simulation import SkeletonTestSimulation


class TestRunMetricAggregator(SkeletonTestSimulation):
    """Test the run_metric_aggregator script."""

    def test_run_metric_aggregator_without_challenges(self) -> None:
        """Sanity test to run metric_aggregator script without any challenges."""
        # Run simulation
        with initialize_config_dir(config_dir=self.config_path):
            # Run without metrics
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    '+simulation=open_loop_boxes',
                    'experiment_name=simulation_metric_aggregator_test',
                ],
            )
            run_simulation(cfg)
            exp_output_dir = deepcopy(cfg.output_dir)

        # Run metric aggregator
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=METRIC_AGGREGATOR_CONFIG_NAME,
                overrides=[
                    f'output_dir={exp_output_dir}',
                    "scenario_metric_paths=[]",
                    "metric_aggregator=[default_weighted_average]",
                    "challenges=[]",
                ],
            )
            run_metric_aggregator(cfg)

            # Check file output
            metric_aggregator_output = Path(cfg.aggregator_save_path)
            aggregator_output_file_length = len(list(metric_aggregator_output.rglob("*")))
            # One default weighted metric aggregator
            self.assertEqual(aggregator_output_file_length, 1)


if __name__ == '__main__':
    unittest.main()
