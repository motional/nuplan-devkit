import os
import tempfile
import unittest
from pathlib import Path

import ray


class SkeletonTestTrain(unittest.TestCase):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def setUp(self) -> None:
        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/training/')

        # Todo: Investigate pkg in hydra
        # Since we are not using the default config in this test, we need to specify the Hydra search path in the
        # compose API override, otherwise the Jenkins build fails because bazel cannot find the simulation config file.
        common_dir = "file://" + os.path.join(main_path, '..', 'config', 'common')
        experiment_dir = "file://" + os.path.join(main_path, '..', 'experiments')
        self.search_path = f'hydra.searchpath=[{common_dir}, {experiment_dir}]'

        # Output directory
        self.tmp_dir = tempfile.TemporaryDirectory()

        # Overrides
        self.default_overrides = [
            'log_config=false',
            'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=100',
            'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.05',
            'lightning.trainer.params.max_epochs=1',
            'lightning.trainer.params.check_val_every_n_epoch=1',
            'lightning.trainer.params.limit_train_batches=1',
            'lightning.trainer.params.limit_val_batches=1',
            'lightning.trainer.params.limit_test_batches=1',
            'data_loader.params.batch_size=2',
            'data_loader.params.num_workers=2',
            'data_loader.params.pin_memory=false',
            f"group={self.tmp_dir.name}",
            f"cache_dir={self.tmp_dir.name}/cache",
            'cleanup_cache=True'
        ]

    def tearDown(self) -> None:

        # Remove experimental dir
        if Path(self.tmp_dir.name).exists():
            self.tmp_dir.cleanup()

        # Stop ray
        if ray.is_initialized():
            ray.shutdown()
