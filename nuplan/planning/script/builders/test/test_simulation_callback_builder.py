import unittest
from pathlib import Path
from unittest import mock

from hypothesis import given
from hypothesis import strategies as st
from omegaconf import DictConfig

from nuplan.planning.script.builders.simulation_callback_builder import (
    build_callbacks_worker,
    build_simulation_callbacks,
)
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.callback.serialization_callback import SerializationCallback
from nuplan.planning.simulation.callback.timing_callback import TimingCallback
from nuplan.planning.simulation.callback.visualization_callback import VisualizationCallback
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor


class TestSimulationCallbackBuilder(unittest.TestCase):
    """Unit tests for functions in simulation_callback_builder.py."""

    mock_cpu_node_count = 4

    @staticmethod
    def _generate_mock_build_callbacks_worker_config(
        number_of_cpus_allocated_per_simulation: int = 1,
        max_callback_workers: int = 1,
        disable_callback_parallelization: bool = False,
    ) -> DictConfig:
        """
        Utility function to generate a mocked callback worker configuration with Sequential worker type. Parameters are
        used directly as the config values.
        """
        return DictConfig(
            {
                "worker": {
                    "_target_": "nuplan.planning.utils.multithreading.worker_sequential.Sequential",
                },
                "number_of_cpus_allocated_per_simulation": number_of_cpus_allocated_per_simulation,
                "max_callback_workers": max_callback_workers,
                "disable_callback_parallelization": disable_callback_parallelization,
            }
        )

    @staticmethod
    def _calculate_expected_number_of_threads(max_callback_workers: int) -> int:
        """
        Utility function to calculate the expected number of threads available to the workers. The calculation is based on
        the current build_callbacks_worker implementation.
        :param max_callback_workers: Config value passed from Sequential worker config.
        """
        # 1 constant -> simplification of build_callbacks_worker logic
        return min(TestSimulationCallbackBuilder.mock_cpu_node_count - 1, max_callback_workers)

    @given(
        number_of_cpus_allocated_per_simulation=st.one_of(st.none(), st.just(1)),
        max_callback_workers=st.integers(min_value=1),
    )
    def test_build_callbacks_worker_nominal(
        self, number_of_cpus_allocated_per_simulation: int, max_callback_workers: int
    ) -> None:
        """Tests the nominal case of build_callbacks_worker."""
        with mock.patch(
            "nuplan.planning.utils.multithreading.worker_pool.WorkerResources.current_node_cpu_count",
            return_value=self.mock_cpu_node_count,
        ):
            mock_config = TestSimulationCallbackBuilder._generate_mock_build_callbacks_worker_config(
                number_of_cpus_allocated_per_simulation=number_of_cpus_allocated_per_simulation,
                max_callback_workers=max_callback_workers,
            )

            worker_pool = build_callbacks_worker(mock_config)

            expected_number_of_threads = TestSimulationCallbackBuilder._calculate_expected_number_of_threads(
                max_callback_workers
            )
            self.assertEqual(worker_pool.number_of_threads, expected_number_of_threads)
            self.assertTrue(isinstance(worker_pool, SingleMachineParallelExecutor))

    @given(
        number_of_cpus_allocated_per_simulation=st.one_of(st.integers(max_value=0), st.integers(min_value=2)),
        max_callback_workers=st.integers(min_value=1),
    )
    def test_build_callbacks_worker_edge_case_invalid_cpus_allocated(
        self, number_of_cpus_allocated_per_simulation: int, max_callback_workers: int
    ) -> None:
        """Tests an edge case of build_callbacks_worker, where an invalid cpu allocation setting is passed."""
        mock_config = TestSimulationCallbackBuilder._generate_mock_build_callbacks_worker_config(
            number_of_cpus_allocated_per_simulation=number_of_cpus_allocated_per_simulation,
            max_callback_workers=max_callback_workers,
            disable_callback_parallelization=False,
        )
        with self.assertRaises(ValueError):
            build_callbacks_worker(mock_config)

    @given(
        number_of_cpus_allocated_per_simulation=st.one_of(st.none(), st.just(1)),
        max_callback_workers=st.integers(min_value=1),
    )
    def test_build_callbacks_worker_edge_cases(
        self, number_of_cpus_allocated_per_simulation: int, max_callback_workers: int
    ) -> None:
        """Tests other edge cases of build_callbacks_worker."""
        # Should return None when target type is not Sequential
        mock_config = TestSimulationCallbackBuilder._generate_mock_build_callbacks_worker_config(
            number_of_cpus_allocated_per_simulation=number_of_cpus_allocated_per_simulation,
            max_callback_workers=max_callback_workers,
            disable_callback_parallelization=False,
        )
        mock_config.worker._target_ = (
            "nuplan.planning.utils.multithreading.worker_parallel.SingleMachineParallelExecutor"
        )

        worker_pool = build_callbacks_worker(mock_config)
        self.assertIsNone(worker_pool)

        # Should return None when disable_callback_parallelization is True
        mock_config = TestSimulationCallbackBuilder._generate_mock_build_callbacks_worker_config(
            number_of_cpus_allocated_per_simulation=number_of_cpus_allocated_per_simulation,
            max_callback_workers=max_callback_workers,
            disable_callback_parallelization=True,
        )
        worker_pool = build_callbacks_worker(mock_config)
        self.assertIsNone(worker_pool)

    def test_build_simulation_callbacks_serialization_callback(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed SerializationCallback config.
        """
        mock_config = DictConfig(
            {
                "callback": {
                    "serialization_callback": {
                        "_target_": "nuplan.planning.simulation.callback.serialization_callback.SerializationCallback",
                        "folder_name": "mock_folder",
                        "serialization_type": "pickle",
                        "serialize_into_single_file": False,
                    }
                }
            }
        )

        callbacks = build_simulation_callbacks(mock_config, Path("/tmp/mock_dir"))
        expected_serialization_callback, *_ = callbacks

        self.assertEqual(1, len(callbacks))
        self.assertTrue(isinstance(expected_serialization_callback, SerializationCallback))

    def test_build_simulation_callbacks_timing_callback(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed TimingCallback config.
        """
        mock_config = DictConfig(
            {
                "callback": {
                    "timing_callback": {
                        "_target_": "nuplan.planning.simulation.callback.timing_callback.TimingCallback",
                    }
                }
            }
        )

        callbacks = build_simulation_callbacks(mock_config, Path("/tmp/mock_dir"))
        expected_timing_callback, *_ = callbacks

        self.assertEqual(1, len(callbacks))
        self.assertTrue(isinstance(expected_timing_callback, TimingCallback))

    def test_build_simulation_callbacks_simulation_log_metric_callbacks(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed SimulationLogCallback
        & MetricCallback configurations.
        """
        mock_config = DictConfig(
            {
                "callback": {
                    "simulation_log_callback": {
                        "_target_": "nuplan.planning.simulation.callback.simulation_log_callback.SimulationLogCallback",
                    },
                    "metric_callback": {
                        "_target_": "nuplan.planning.simulation.callback.metric_callback.MetricCallback",
                    },
                }
            }
        )

        callbacks = build_simulation_callbacks(mock_config, Path("/tmp/mock_dir"))

        self.assertEqual(0, len(callbacks))  # simulation log & metric callbacks are not supported

    def test_build_simulation_callbacks_multi_callback(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed MultiCallback config.
        """
        mock_config = DictConfig(
            {
                "callback": {
                    "multi_callback": {
                        "_target_": "nuplan.planning.simulation.callback.multi_callback.MultiCallback",
                        "callbacks": [],
                    },
                }
            }
        )

        callbacks = build_simulation_callbacks(mock_config, Path("/tmp/mock_dir"))
        expected_multi_callback, *_ = callbacks

        self.assertEqual(1, len(callbacks))
        self.assertTrue(isinstance(expected_multi_callback, MultiCallback))

    def test_build_simulation_callbacks_visualization_callback(self) -> None:
        """
        Tests that build_simulation_callbacks returns the expected result when passed MultiCallback config.
        """
        mock_config = DictConfig(
            {
                "callback": {
                    "visualization_callback": {
                        "_target_": "nuplan.planning.simulation.callback.visualization_callback.VisualizationCallback",
                        "renderer": {},
                    },
                }
            }
        )

        callbacks = build_simulation_callbacks(mock_config, Path("/tmp/mock_dir"))
        expected_visualization_callback, *_ = callbacks

        self.assertEqual(1, len(callbacks))
        self.assertTrue(isinstance(expected_visualization_callback, VisualizationCallback))


if __name__ == "__main__":
    unittest.main()
