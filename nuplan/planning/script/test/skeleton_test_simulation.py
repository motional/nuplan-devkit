import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

import ray


class SkeletonTestSimulation(unittest.TestCase):
    """
    Test main simulation entry point using the same config.
    """

    def __init__(self, *args: Any, main_path: Optional[Path] = None, **kwargs: Any):
        """
        Constructor for the class SkeletonTestSimulation.
        :param args: Arguments.
        :param main_path: The main path to search hydra config paths from.
        :param kwargs: Keyword arguments.
        """
        super(SkeletonTestSimulation, self).__init__(*args, **kwargs)
        self._main_path = main_path

    def setUp(self) -> None:
        """Set up basic configs."""
        self._main_path = self._main_path if self._main_path else Path(os.path.realpath(__file__)).parent
        self.config_path = str(self._main_path.parent / "config/simulation/")
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
            'worker=sequential',
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
