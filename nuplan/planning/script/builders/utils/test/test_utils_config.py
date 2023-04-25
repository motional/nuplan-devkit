import os
import unittest
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory, mkstemp
from typing import Optional
from unittest.mock import patch

from hypothesis import example, given, settings
from hypothesis import strategies as st
from omegaconf import DictConfig, ListConfig

from nuplan.common.utils.test_utils.patch import patch_with_validation
from nuplan.planning.script.builders.utils.utils_config import (
    get_num_gpus_used,
    update_config_for_nuboard,
    update_config_for_simulation,
    update_config_for_training,
)


class TestUtilsConfig(unittest.TestCase):
    """Tests for the non-distributed training functions in utils_config.py."""

    specific_world_size = 4

    @staticmethod
    def _generate_mock_training_config() -> DictConfig:
        """
        Returns a mock training configuration with sensible default values.
        :return: DictConfig representing the training configuration.
        """
        return DictConfig(
            {
                "log_config": True,
                "experiment": "mock_experiment_name",
                "group": "mock_group_name",
                "cache": {
                    "cleanup_cache": False,
                    "cache_path": None,
                },
                "data_loader": {
                    "params": {
                        "num_workers": None,
                    },
                },
                "lightning": {
                    "trainer": {
                        "params": {
                            "gpus": None,
                            "accelerator": None,
                            "precision": None,
                        },
                        "overfitting": {"enable": False},
                    },
                },
                "gpu": False,
            }
        )

    @staticmethod
    def _generate_mock_simulation_config() -> DictConfig:
        """
        Returns a mock simulation configuration with sensible default values.
        :return: DictConfig representing the simulation configuration.
        """
        return DictConfig(
            {
                "log_config": True,
                "experiment": "mock_experiment_name",
                "group": "mock_group_name",
                "callback": {
                    "timing_callback": {
                        "_target_": "nuplan.planning.simulation.callback.timing_callback.TimingCallback",
                    },
                    "simulation_log_callback": {
                        "_target_": "nuplan.planning.simulation.callback.simulation_log_callback.SimulationLogCallback",
                    },
                    "metric_callback": {
                        "_target_": "nuplan.planning.simulation.callback.metric_callback.MetricCallback",
                    },
                },
            }
        )

    @staticmethod
    def _patch_return_false() -> bool:
        """A patch function that will always return False."""
        return False

    @staticmethod
    def _patch_return_true() -> bool:
        """A patch function that will always return True."""
        return True

    @given(cache_path=st.one_of(st.none(), st.just("s3://bucket/key")))
    @settings(deadline=None)
    def test_update_config_for_training_cache_path_none_or_s3(self, cache_path: Optional[str]) -> None:
        """
        Tests the behavior of update_config_for_training when the supplied cfg.cache.cache_path is either
        None or an S3 path.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        mock_config.cache.cache_path = cache_path

        with TemporaryDirectory() as tmp_dir:
            with patch_with_validation("torch.cuda.is_available", TestUtilsConfig._patch_return_false):
                update_config_for_training(mock_config)

            self.assertTrue(Path(tmp_dir).exists())  # the existing directory is untouched

    def test_update_config_for_training_cache_path_local_non_existing(self) -> None:
        """
        Tests the behavior of update_config_for_training when the supplied cfg.cache.cache_path doesn't exist yet.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()

        with TemporaryDirectory() as tmp_dir:
            mock_config.cache.cache_path = tmp_dir
            rmtree(tmp_dir)  # delete the directory, so it's nonexistent when the function is called

            with patch_with_validation("torch.cuda.is_available", TestUtilsConfig._patch_return_false):
                update_config_for_training(mock_config)

            self.assertTrue(Path(tmp_dir).exists())  # the directory was created

    def test_update_config_for_training_cache_path_local_cleanup(self) -> None:
        """
        Tests the behavior of update_config_for_training when the supplied cfg.cache.cache_path exists and
        cleanup_cache is requested.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()

        with TemporaryDirectory() as tmp_dir:
            _, tmp_file = mkstemp(dir=tmp_dir)
            mock_config.cache.cache_path = tmp_dir
            mock_config.cache.cleanup_cache = True

            self.assertTrue(Path(tmp_file).exists())  # check that the temp file initially exists

            with patch_with_validation("torch.cuda.is_available", TestUtilsConfig._patch_return_false):
                update_config_for_training(mock_config)

            self.assertFalse(Path(tmp_file).exists())  # check that the file was cleaned up
            self.assertTrue(Path(tmp_dir).exists())  # check that the directory still exists

        self.assertFalse(Path(tmp_dir).exists())

    def test_update_config_for_training_overfitting(self) -> None:
        """
        Tests the behavior of update_config_for_training in regard to overfitting configurations.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()

        # Case #1: num_workers shouldn't be modified when overfitting is disabled
        num_workers = 32  # selected arbitrarily
        mock_config.data_loader.params.num_workers = num_workers
        mock_config.lightning.trainer.overfitting.enable = False

        with patch_with_validation("torch.cuda.is_available", TestUtilsConfig._patch_return_false):
            update_config_for_training(mock_config)

        self.assertEqual(num_workers, mock_config.data_loader.params.num_workers)

        # Case #2: num_workers should be set to 0 when overfitting is enabled
        mock_config.lightning.trainer.overfitting.enable = True

        with patch_with_validation("torch.cuda.is_available", TestUtilsConfig._patch_return_false):
            update_config_for_training(mock_config)

        self.assertEqual(0, mock_config.data_loader.params.num_workers)

    @given(is_gpu_enabled=st.booleans(), is_cuda_available=st.booleans())
    @settings(deadline=None)
    def test_update_config_for_training_gpu(self, is_gpu_enabled: bool, is_cuda_available: bool) -> None:
        """
        Tests the behavior of update_config_for_training in regard to gpu configurations.
        """
        invalid_value = -99  # test-specific constant
        cuda_patch = TestUtilsConfig._patch_return_true if is_cuda_available else TestUtilsConfig._patch_return_false

        def get_expected_gpu_config(gpu_enabled: bool, cuda_available: bool) -> Optional[int]:
            return -1 if gpu_enabled and cuda_available else None

        def get_expected_accelerator_config(gpu_enabled: bool, cuda_available: bool) -> Optional[int]:
            return invalid_value if gpu_enabled and cuda_available else None

        def get_expected_precision_config(gpu_enabled: bool, cuda_available: bool) -> Optional[int]:
            return invalid_value if gpu_enabled and cuda_available else 32

        mock_config = TestUtilsConfig._generate_mock_training_config()
        mock_config.gpu = is_gpu_enabled

        # initialize values to test with invalid_value
        mock_config.lightning.trainer.params.gpus = invalid_value
        mock_config.lightning.trainer.params.accelerator = invalid_value
        mock_config.lightning.trainer.params.precision = invalid_value

        with patch_with_validation("torch.cuda.is_available", cuda_patch):
            update_config_for_training(mock_config)

        self.assertEqual(
            get_expected_gpu_config(is_gpu_enabled, is_cuda_available), mock_config.lightning.trainer.params.gpus
        )
        self.assertEqual(
            get_expected_accelerator_config(is_gpu_enabled, is_cuda_available),
            mock_config.lightning.trainer.params.accelerator,
        )
        self.assertEqual(
            get_expected_precision_config(is_gpu_enabled, is_cuda_available),
            mock_config.lightning.trainer.params.precision,
        )

    @given(max_number_of_workers=st.one_of(st.none(), st.just(0)))
    @settings(deadline=None)
    def test_update_config_for_simulation_falsy_max_number_of_workers(self, max_number_of_workers: int) -> None:
        """
        Tests that update_config_for_simulation works as expected.
        When max number of workers is falsy, timing_callback won't be removed
        """
        mock_config = TestUtilsConfig._generate_mock_simulation_config()
        mock_config.max_number_of_workers = max_number_of_workers

        update_config_for_simulation(mock_config)

        self.assertEqual(3, len(mock_config.callback))

    @given(max_number_of_workers=st.integers(min_value=1))
    @settings(deadline=None)
    def test_update_config_for_simulation_truthy_max_number_of_workers(self, max_number_of_workers: int) -> None:
        """
        Tests that update_config_for_simulation works as expected. When max number of workers is truthy, a new
        `callbacks` entry will be added. The values are taken from `callback` with timing_callback target removed.
        """
        mock_config = TestUtilsConfig._generate_mock_simulation_config()
        mock_config.max_number_of_workers = max_number_of_workers

        update_config_for_simulation(mock_config)

        self.assertEqual(3, len(mock_config.callback))
        self.assertEqual(2, len(mock_config.callbacks))
        callbacks_targets = [callback["_target_"] for callback in mock_config.callbacks]
        self.assertNotIn("nuplan.planning.simulation.callback.timing_callback.TimingCallback", callbacks_targets)

    def test_update_config_for_nuboard(self) -> None:
        """Tests that update_config_for_nuboard works as expected."""
        mock_config = DictConfig(
            {
                "log_config": True,
            }
        )

        # Case #1 - simulation_path is None
        mock_config.simulation_path = None

        update_config_for_nuboard(mock_config)

        self.assertIsNotNone(mock_config.simulation_path)
        self.assertEqual(0, len(mock_config.simulation_path))  # type: ignore

        # Case #2 - simulation_path is an instance of list
        simulation_path_list = ["/mock/path", "/to/somewhere"]
        mock_config.simulation_path = simulation_path_list

        update_config_for_nuboard(mock_config)

        self.assertEqual(simulation_path_list, mock_config.simulation_path)

        # Case #3 - simulation_path is an instance of ListConfig
        simulation_path_list_config = ListConfig(element_type=str, content=["/mock/path", "/to/somewhere"])
        mock_config.simulation_path = simulation_path_list_config

        update_config_for_nuboard(mock_config)

        self.assertEqual(simulation_path_list_config, mock_config.simulation_path)

        # Case #4 - simulation_path is a string
        simulation_path = "/mock/path"
        mock_config.simulation_path = simulation_path

        update_config_for_nuboard(mock_config)

        expected_simulation_path_list = [simulation_path]
        self.assertEqual(expected_simulation_path_list, mock_config.simulation_path)

    @patch.dict(os.environ, {"WORLD_SIZE": str(specific_world_size)}, clear=True)
    def test_get_num_gpus_used_from_world_size(self) -> None:
        """
        Tests that that get_num_gpus_used works as expected. When WORLD_SIZE is set to a specific value, the function
        will simply return that value.
        """
        mock_config = DictConfig({})

        num_gpus = get_num_gpus_used(mock_config)

        self.assertEqual(self.specific_world_size, num_gpus)

    @given(
        num_gpus_config=st.integers(min_value=-1),
        cuda_device_count=st.integers(min_value=0),
        num_nodes=st.integers(min_value=1),
    )
    @example(num_gpus_config=-1, cuda_device_count=2, num_nodes=2)
    @settings(deadline=None)
    def test_get_num_gpus_used_from_config(self, num_gpus_config: int, cuda_device_count: int, num_nodes: int) -> None:
        """
        Tests that that get_num_gpus_used works as expected when WORLD_SIZE environment variable is not set.
        """

        def patch_get_cuda_device_count() -> int:
            return cuda_device_count

        with (
            patch.dict(os.environ, {"NUM_NODES": str(num_nodes)}, clear=True),
            patch_with_validation("torch.cuda.device_count", patch_get_cuda_device_count),
        ):
            mock_config = TestUtilsConfig._generate_mock_training_config()
            mock_config.lightning.trainer.params.gpus = num_gpus_config

            num_gpus = get_num_gpus_used(mock_config)
            expected_num_gpus = num_gpus_config if num_gpus_config != -1 else cuda_device_count * num_nodes

            self.assertEqual(expected_num_gpus, num_gpus)

    def test_get_num_gpus_used_invalid_config(self) -> None:
        """
        Tests that that get_num_gpus_used raises a RuntimeError when WORLD_SIZE environment variable is not set and
        a string is passed as the value of mock_config.lightning.trainer.params.gpus.
        """
        mock_config = TestUtilsConfig._generate_mock_training_config()
        mock_config.lightning.trainer.params.gpus = "1"

        with self.assertRaises(RuntimeError):
            get_num_gpus_used(mock_config)


if __name__ == "__main__":
    unittest.main()
