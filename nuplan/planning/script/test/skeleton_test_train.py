import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Optional

import ray


class SkeletonTestTrain(unittest.TestCase):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def __init__(self, *args: Any, additional_paths: Optional[List[str]] = None, **kwargs: Any):
        """
        Constructor for the class SkeletonTestTrain
        :param args: Arguments.
        :param additional_paths: Any additional paths needed for hydra
        :param kwargs: Keyword arguments.
        """
        super(SkeletonTestTrain, self).__init__(*args, **kwargs)
        self.additional_paths = additional_paths if additional_paths is not None else []

    def setUp(self) -> None:
        """Set up basic config."""
        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/training/')

        # TODO: Investigate pkg in hydra
        # Since we are not using the default config in this test, we need to specify the Hydra search path in the
        # compose API override, otherwise the Jenkins build fails because bazel cannot find the simulation config file.
        common_dir = 'file://' + os.path.join(main_path, '..', 'config', 'common')
        experiment_dir = 'file://' + os.path.join(main_path, '..', 'experiments')
        training_dir = "file://" + os.path.join(main_path, '..', 'config', 'training')
        self.search_path = f'hydra.searchpath=[{common_dir}, {experiment_dir}, {training_dir}'

        # Append additional paths to the search path
        self.search_path += ', '.join(['', *self.additional_paths]) + ']'

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
