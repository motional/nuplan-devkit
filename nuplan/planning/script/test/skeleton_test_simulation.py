import os
import tempfile
import unittest
from pathlib import Path

import ray


class SkeletonTestSimulation(unittest.TestCase):
    """
    Test main simulation entry point using the same config.
    """

    def setUp(self) -> None:
        """Set up basic configs."""
        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/simulation/')
        # Since we are not using the default config in this test, we need to specify the Hydra search path in the
        # compose API override, otherwise the Jenkins build fails because bazel cannot find the simulation config file.
        self.tmp_dir = tempfile.TemporaryDirectory()
        # Default hydra overrides for quick unit testing
        self.default_overrides = [
            'log_config=false',
            'scenario_builder=nuplan_mini',
            'planner=simple_planner',
            'scenario_filter=one_of_each_scenario_type',
            'scenario_filter.limit_total_scenarios=2',
            'exit_on_failure=true',
            f'group={self.tmp_dir.name}',
            'job_name=test_simulation',
            'output_dir=${group}/${experiment}',
        ]

    def tearDown(self) -> None:
        """Clean up."""
        if Path(self.tmp_dir.name).exists():
            self.tmp_dir.cleanup()

        # Stop ray
        if ray.is_initialized():
            ray.shutdown()
