import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

import ray


class SkeletonTestTrain(unittest.TestCase):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def __init__(self, *args: Any, main_path: Optional[str] = None, **kwargs: Any):
        """
        Constructor for the class SkeletonTestTrain
        :param args: Arguments.
        :param additional_paths: Any additional paths needed for hydra
        :param kwargs: Keyword arguments.
        """
        super(SkeletonTestTrain, self).__init__(*args, **kwargs)
        self._main_path = main_path

    def setUp(self) -> None:
        """Set up basic config."""
        if not self._main_path:
            self._main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(self._main_path, '../config/training/')

        # Output directory
        self.tmp_dir = tempfile.TemporaryDirectory()

        # Overrides
        self.default_overrides = [
            'log_config=false',
            'worker=sequential',
            'scenario_filter.limit_total_scenarios=30',
            'lightning.trainer.params.max_epochs=1',
            'lightning.trainer.params.check_val_every_n_epoch=1',
            'lightning.trainer.params.limit_train_batches=1',
            'lightning.trainer.params.limit_val_batches=1',
            'lightning.trainer.params.limit_test_batches=1',
            'data_loader.params.batch_size=2',
            'data_loader.params.num_workers=2',
            'data_loader.params.pin_memory=false',
            f'group={self.tmp_dir.name}',
            f'cache.cache_path={self.tmp_dir.name}/cache',
            'cache.cleanup_cache=True',
            'output_dir=${group}/${experiment}',
        ]

    def tearDown(self) -> None:
        """Clean up."""
        # Remove experimental dir
        if Path(self.tmp_dir.name).exists():
            self.tmp_dir.cleanup()

        # Stop ray
        if ray.is_initialized():
            ray.shutdown()
