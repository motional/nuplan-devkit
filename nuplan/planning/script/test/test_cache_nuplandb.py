import asyncio
import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from nuplan.common.utils.s3_utils import list_files_in_s3_directory, split_s3_path
from nuplan.common.utils.test_utils.mock_s3_utils import create_mock_bucket, mock_async_s3, set_mock_object_from_aws
from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain


class TestCache(SkeletonTestTrain):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def setUp(self) -> None:
        """
        Set up test attributes.
        """
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.local_cache_path = f'{self.tmp_dir.name}/cache'
        self.s3_cache_path = "s3://test-bucket/nuplan_tests/test_cache_nuplandb"
        self.test_args = [
            '+training=training_raster_model',
            'scenario_builder=nuplan_mini',
            'splitter=nuplan',
            f'group={self.tmp_dir.name}',
        ]

    def tearDown(self) -> None:
        """
        Cleanup after each test.
        """
        self.tmp_dir.cleanup()

    @unittest.skip("Skip in CI until issue is resolved")
    def test_cache_dataset_s3(self) -> None:
        """
        Tests dataset caching with mocked S3.
        """
        s3_bucket, s3_key = split_s3_path(self.s3_cache_path)
        set_mock_object_from_aws(
            Path("nuplan-v1.1/maps/us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg"), "nuplan-production"
        )

        with mock_async_s3():
            asyncio.run(create_mock_bucket(s3_bucket))

            # Create cache
            with initialize_config_dir(config_dir=self.config_path):
                cfg = compose(
                    config_name=CONFIG_NAME,
                    overrides=[
                        *self.default_overrides,
                        *self.test_args,
                        'scenario_filter.limit_total_scenarios=10',
                        'py_func=cache',
                        f'cache.cache_path={self.s3_cache_path}',
                        'cache.force_feature_computation=True',
                    ],
                )
                main(cfg)

            self.assertTrue(len(list_files_in_s3_directory(s3_key, s3_bucket)) > 0)

            # Use cache without loading the dataset
            with initialize_config_dir(config_dir=self.config_path):
                cfg = compose(
                    config_name=CONFIG_NAME,
                    overrides=[
                        *self.default_overrides,
                        *self.test_args,
                        'py_func=train',
                        'scenario_filter.limit_total_scenarios=10',
                        'cache.cleanup_cache=false',
                        'cache.use_cache_without_dataset=true',
                        f'cache.cache_path={self.s3_cache_path}',
                    ],
                )
                main(cfg)

            # Use cache with loading the dataset
            with initialize_config_dir(config_dir=self.config_path):
                cfg = compose(
                    config_name=CONFIG_NAME,
                    overrides=[
                        *self.default_overrides,
                        *self.test_args,
                        'py_func=train',
                        'scenario_filter.limit_total_scenarios=10',
                        'cache.cleanup_cache=false',
                        'cache.use_cache_without_dataset=false',
                        f'cache.cache_path={self.s3_cache_path}',
                    ],
                )
                main(cfg)

    def test_cache_dataset_local(self) -> None:
        """
        Tests local dataset caching.
        """
        # Create cache
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    *self.test_args,
                    'py_func=cache',
                    f'cache.cache_path={self.local_cache_path}',
                ],
            )
            main(cfg)

        self.assertTrue(any(Path(self.local_cache_path).iterdir()))

        # Use cache without loading the dataset
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    *self.test_args,
                    'py_func=train',
                    'cache.cleanup_cache=false',
                    'cache.use_cache_without_dataset=true',
                    f'cache.cache_path={self.local_cache_path}',
                ],
            )
            main(cfg)

        # Use cache with loading the dataset
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    *self.test_args,
                    'py_func=train',
                    'cache.cleanup_cache=false',
                    'cache.use_cache_without_dataset=false',
                    f'cache.cache_path={self.local_cache_path}',
                ],
            )
            main(cfg)

    def test_profiling(self) -> None:
        """Test that profiling gets generated."""
        # Create cache
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    *self.test_args,
                    'py_func=cache',
                    'enable_profiling=True',
                    f'cache.cache_path={self.local_cache_path}',
                ],
            )
            main(cfg)

        self.assertTrue(Path(self.local_cache_path).rglob("caching.html"))


if __name__ == '__main__':
    unittest.main()
