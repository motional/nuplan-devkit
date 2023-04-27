import gc
import unittest
from typing import Callable, Generator, List, Optional, Set, Tuple, Type, Union
from unittest.mock import Mock, patch

import guppy
import mock
import numpy as np
import PIL.Image as PilImg

import nuplan.database.nuplan_db.image as ImageDBRow
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.utils.test_utils.interface_validation import assert_class_properly_implements_interface
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplan_db_utils import SensorDataSource, get_lidarpc_sensor_data
from nuplan.database.nuplan_db.sensor_data_table_row import SensorDataTableRow
from nuplan.database.nuplan_db.test.minimal_db_test_utils import int_to_str_token, str_token_to_int
from nuplan.database.utils.image import Image
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.planning.simulation.observation.observation_type import CameraChannel, LidarChannel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

CAMERA_OFFSET = 123
TEST_PATH = "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario"
TEST_PATH_UTILS = "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils"


class TestNuPlanScenario(unittest.TestCase):
    """
    Tests scenario for NuPlan
    """
    def _make_test_scenario(
        self,
    ) -> NuPlanScenario:
        """
        Creates a sample scenario to use for testing.
        """
        return NuPlanScenario(
            data_root="data_root/",
            log_file_load_path="data_root/log_name.db",
            initial_lidar_token=int_to_str_token(1234),
            initial_lidar_timestamp=2345,
            scenario_type="scenario_type",
            map_root="map_root",
            map_version="map_version",
            map_name="map_name",
            scenario_extraction_info=ScenarioExtractionInfo(
                scenario_name="scenario_name", scenario_duration=20, extraction_offset=1, subsample_ratio=0.5
            ),
            ego_vehicle_parameters=get_pacifica_parameters(),
            sensor_root="sensor_root",
        )

    def _get_sampled_sensor_tokens_in_time_window_patch(
        self,
        expected_log_file: str,
        expected_sensor_data_source: SensorDataSource,
        expected_start_timestamp: int,
        expected_end_timestamp: int,
        expected_subsample_step: int,
    ) -> Callable[[str, SensorDataSource, int, int, int], Generator[str, None, None]]:
        """
        Creates a patch for the get_sampled_lidarpc_tokens_in_time_window function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_start_timestamp: The expected start timestamp with which the function is expected to be called.
        :param expected_end_timestamp: The expected end timestamp with which the function is expected to be called.
        :param expected_subsample_step: The expected subsample step with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(
            actual_log_file: str,
            actual_sensor_data_source: SensorDataSource,
            actual_start_timestamp: int,
            actual_end_timestamp: int,
            actual_subsample_step: int,
        ) -> Generator[str, None, None]:
            """
            The patch function for get_sampled_lidarpc_tokens_in_time_window.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_start_timestamp, actual_start_timestamp)
            self.assertEqual(expected_end_timestamp, actual_end_timestamp)
            self.assertEqual(expected_subsample_step, actual_subsample_step)

            num_tokens = int((expected_end_timestamp - expected_start_timestamp) / (expected_subsample_step * 1e6))
            for token in range(num_tokens):
                yield int_to_str_token(token)

        return fxn

    def _get_download_file_if_necessary_patch(
        self, expected_data_root: str, expected_log_file_load_path: str
    ) -> Callable[[str, str], str]:
        """
        Creates a patch for the download_file_if_necessary function that validates the arguments.
        :param expected_data_root: The data_root with which the function is expected to be called.
        :param expected_log_file_load_path: The log_file_load_path with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_data_root: str, actual_log_file_load_path: str) -> str:
            """
            The generated patch function.
            """
            self.assertEqual(expected_data_root, actual_data_root)
            self.assertEqual(expected_log_file_load_path, actual_log_file_load_path)

            return actual_log_file_load_path

        return fxn

    def _get_sensor_data_from_sensor_data_tokens_from_db_patch(
        self,
        expected_log_file: str,
        expected_sensor_data_source: SensorDataSource,
        expected_sensor_class: Type[SensorDataTableRow],
        expected_tokens: List[str],
    ) -> Callable[
        [str, SensorDataSource, Type[SensorDataTableRow], List[str]], Generator[SensorDataTableRow, None, None]
    ]:
        """
        Creates a patch for the get_images_from_lidar_tokens_patch function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_sensor_data_source: The sensor source with which the function is expected to be called.
        :param expected_sensor_class: The sensor class with which the function is expected to be called.
        :param expected_tokens: The tokens with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(
            actual_log_file: str,
            actual_sensor_data_source: SensorDataSource,
            actual_sensor_class: Type[SensorDataTableRow],
            actual_tokens: List[str],
        ) -> Generator[SensorDataTableRow, None, None]:
            """
            The patch function for get_sensor_data_from_sensor_data_tokens_from_db.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_sensor_class, actual_sensor_class)
            self.assertEqual(expected_tokens, actual_tokens)
            lidar_token = actual_tokens[0]
            if expected_sensor_class == LidarPc:
                yield LidarPc(
                    token=lidar_token,
                    next_token=lidar_token,
                    prev_token=lidar_token,
                    ego_pose_token=lidar_token,
                    lidar_token=lidar_token,
                    scene_token=lidar_token,
                    filename=f"lidar_{lidar_token}",
                    timestamp=str_token_to_int(lidar_token),
                )

            elif expected_sensor_class == ImageDBRow.Image:
                camera_token = str_token_to_int(lidar_token) + CAMERA_OFFSET
                yield ImageDBRow.Image(
                    token=int_to_str_token(camera_token),
                    next_token=int_to_str_token(camera_token),
                    prev_token=int_to_str_token(camera_token),
                    ego_pose_token=int_to_str_token(camera_token),
                    camera_token=int_to_str_token(camera_token),
                    filename_jpg=f"image_{camera_token}",
                    timestamp=camera_token,
                    channel=CameraChannel.CAM_R0.value,
                )
            else:
                self.fail(f"Unexpected type: {expected_sensor_class}.")

        return fxn

    def _load_point_cloud_patch(
        self, expected_lidar_pc: LidarPc, expected_local_store: LocalStore, expected_s3_store: S3Store
    ) -> Callable[[LidarPc, LocalStore, S3Store], LidarPointCloud]:
        """
        Creates a patch for the _load_point_cloud function that validates the arguments.
        :param expected_lidar_pc: The lidar pc with which the function is expected to be called.
        :param expected_local_store: The LocalStore with which the function is expected to be called.
        :param expected_s3_store: The S3Store with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_lidar_pc: LidarPc, actual_local_store: LocalStore, actual_s3_store: S3Store) -> LidarPointCloud:
            """
            The patch function for load_point_cloud.
            """
            self.assertEqual(expected_lidar_pc, actual_lidar_pc)
            self.assertEqual(expected_local_store, actual_local_store)
            self.assertEqual(expected_s3_store, actual_s3_store)
            return LidarPointCloud(np.eye(3))

        return fxn

    def _load_image_patch(
        self, expected_local_store: LocalStore, expected_s3_store: S3Store
    ) -> Callable[[ImageDBRow.Image, LocalStore, S3Store], Image]:
        """
        Creates a patch for the _load_image_patch function and validates that argument is an Image object.
        :param expected_local_store: The LocalStore with which the function is expected to be called.
        :param expected_s3_store: The S3Store with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_image: ImageDBRow.Image, actual_local_store: LocalStore, actual_s3_store: S3Store) -> Image:
            """
            The patch function for load_image.
            """
            self.assertEqual(expected_local_store, actual_local_store)
            self.assertEqual(expected_s3_store, actual_s3_store)
            self.assertTrue(isinstance(actual_image, ImageDBRow.Image))
            return Image(PilImg.new('RGB', (500, 500)))

        return fxn

    def _get_images_from_lidar_tokens_patch(
        self,
        expected_log_file: str,
        expected_tokens: List[str],
        expected_channels: List[str],
        expected_lookahead_window_us: int,
        expected_lookback_window_us: int,
    ) -> Callable[[str, List[str], List[str], int, int], Generator[ImageDBRow.Image, None, None]]:
        """
        Creates a patch for the get_images_from_lidar_tokens_patch function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_tokens: The expected tokens with which the function is expected to be called.
        :param expected_channels: The expected channels with which the function is expected to be called.
        :param expected_lookahead_window_us: The expected lookahead window with which the function is expected to be called.
        :param expected_lookahead_window_us: The expected lookback window with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(
            actual_log_file: str,
            actual_tokens: List[str],
            actual_channels: List[str],
            actual_lookahead_window_us: int = 50000,
            actual_lookback_window_us: int = 50000,
        ) -> Generator[ImageDBRow.Image, None, None]:
            """
            The patch function for get_images_from_lidar_tokens.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_tokens, actual_tokens)
            self.assertEqual(expected_channels, actual_channels)
            self.assertEqual(expected_lookahead_window_us, actual_lookahead_window_us)
            self.assertEqual(expected_lookback_window_us, actual_lookback_window_us)

            for camera_token, channel in enumerate(actual_channels):
                if channel != LidarChannel.MERGED_PC.value:
                    yield ImageDBRow.Image(
                        token=int_to_str_token(camera_token),
                        next_token=int_to_str_token(camera_token),
                        prev_token=int_to_str_token(camera_token),
                        ego_pose_token=int_to_str_token(camera_token),
                        camera_token=int_to_str_token(camera_token),
                        filename_jpg=f"image_{camera_token}",
                        timestamp=camera_token,
                        channel=channel,
                    )

        return fxn

    def _get_sampled_lidarpcs_from_db_patch(
        self,
        expected_log_file: str,
        expected_initial_token: str,
        expected_sensor_data_source: SensorDataSource,
        expected_sample_indexes: Union[Generator[int, None, None], List[int]],
        expected_future: bool,
    ) -> Callable[
        [str, str, SensorDataSource, Union[Generator[int, None, None], List[int]], bool], Generator[LidarPc, None, None]
    ]:
        """
        Creates a patch for the get_sampled_lidarpcs_from_db function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_initial_token: The initial token name with which the function is expected to be called.
        :param expected_sensor_data_source: The sensor source with which the function is expected to be called.
        :param expected_sample_indexes: The sample indexes with which the function is expected to be called.
        :param expected_future: The future with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(
            actual_log_file: str,
            actual_initial_token: str,
            actual_sensor_data_source: SensorDataSource,
            actual_sample_indexes: Union[Generator[int, None, None], List[int]],
            actual_future: bool,
        ) -> Generator[LidarPc, None, None]:
            """
            The patch function for get_images_from_lidar_tokens.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_initial_token, actual_initial_token)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_sample_indexes, actual_sample_indexes)
            self.assertEqual(expected_future, actual_future)

            for idx in actual_sample_indexes:
                lidar_token = int_to_str_token(idx)
                yield LidarPc(
                    token=lidar_token,
                    next_token=lidar_token,
                    prev_token=lidar_token,
                    ego_pose_token=lidar_token,
                    lidar_token=lidar_token,
                    scene_token=lidar_token,
                    filename=f"lidar_{lidar_token}",
                    timestamp=str_token_to_int(lidar_token),
                )

        return fxn

    def test_implements_abstract_scenario_interface(self) -> None:
        """
        Tests that NuPlanScenario properly implements AbstractScenario interface.
        """
        assert_class_properly_implements_interface(AbstractScenario, NuPlanScenario)

    def test_token(self) -> None:
        """
        Tests that the token method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
            download_file_patch_fxn,
        ):
            scenario = self._make_test_scenario()
            self.assertEqual(int_to_str_token(1234), scenario.token)

    def test_log_name(self) -> None:
        """
        Tests that the log_name method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
            download_file_patch_fxn,
        ):
            scenario = self._make_test_scenario()
            self.assertEqual("log_name", scenario.log_name)

    def test_scenario_name(self) -> None:
        """
        Tests that the scenario_name method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
            download_file_patch_fxn,
        ):
            scenario = self._make_test_scenario()
            self.assertEqual(int_to_str_token(1234), scenario.scenario_name)

    def test_ego_vehicle_parameters(self) -> None:
        """
        Tests that the ego_vehicle_parameters method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
            download_file_patch_fxn,
        ):
            scenario = self._make_test_scenario()
            self.assertEqual(get_pacifica_parameters(), scenario.ego_vehicle_parameters)

    def test_scenario_type(self) -> None:
        """
        Tests that the scenario_type method works properly
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
            download_file_patch_fxn,
        ):
            scenario = self._make_test_scenario()
            self.assertEqual("scenario_type", scenario.scenario_type)

    def test_database_interval(self) -> None:
        """
        Tests that the database_interval method works properly
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
            download_file_patch_fxn,
        ):
            scenario = self._make_test_scenario()
            self.assertEqual(0.1, scenario.database_interval)

    def test_get_number_of_iterations(self) -> None:
        """
        Tests that the get_number_of_iterations method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
            expected_sensor_data_source=get_lidarpc_sensor_data(),
            expected_start_timestamp=int((1 * 1e6) + 2345),
            expected_end_timestamp=int((21 * 1e6) + 2345),
            expected_subsample_step=2,
        )

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
            download_file_patch_fxn,
        ):
            with mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db",
                lidarpc_tokens_patch_fxn,
            ):
                scenario = self._make_test_scenario()
                self.assertEqual(10, scenario.get_number_of_iterations())

    def test_get_time_point(self) -> None:
        """
        Tests that the get_time_point method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
            expected_sensor_data_source=get_lidarpc_sensor_data(),
            expected_start_timestamp=int((1 * 1e6) + 2345),
            expected_end_timestamp=int((21 * 1e6) + 2345),
            expected_subsample_step=2,
        )

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        for iter_val in [0, 3, 5]:

            def token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch method for get_lidarpc_token_timstamp_from_db that validates the arguments.
                """
                self.assertEqual("data_root/log_name.db", log_file)
                self.assertEqual(
                    SensorDataSource(
                        table='lidar_pc',
                        sensor_table='lidar',
                        sensor_token_column='lidar_token',
                        channel='MergedPointCloud',
                    ),
                    sensor_source,
                )
                self.assertEqual(int_to_str_token(iter_val), token)

                return int(str_token_to_int(iter_val) + 5)

            with mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
                download_file_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db",
                lidarpc_tokens_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.get_sensor_data_token_timestamp_from_db",
                token_timestamp_patch,
            ):
                scenario = self._make_test_scenario()
                self.assertEqual(iter_val + 5, scenario.get_time_point(iter_val).time_us)

    def test_get_tracked_objects_at_iteration(self) -> None:
        """
        Tests that the get_tracked_objects_at_iteration method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
            expected_sensor_data_source=get_lidarpc_sensor_data(),
            expected_start_timestamp=int((1 * 1e6) + 2345),
            expected_end_timestamp=int((21 * 1e6) + 2345),
            expected_subsample_step=2,
        )

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)

        for iter_val in [0, 2, 3]:

            def get_token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch for get_sensor_data_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual("data_root/log_name.db", log_file)
                self.assertEqual(
                    SensorDataSource(
                        table='lidar_pc',
                        sensor_table='lidar',
                        sensor_token_column='lidar_token',
                        channel='MergedPointCloud',
                    ),
                    sensor_source,
                )
                self.assertEqual(int_to_str_token(iter_val), token)

                return int(iter_val * 1e6)

            def tracked_objects_for_token_patch(log_file: str, token: str) -> Generator[TrackedObject, None, None]:
                """
                The patch for get_tracked_objects_for_lidarpc_token that validates the arguments and generates fake data.
                """
                self.assertEqual("data_root/log_name.db", log_file)
                self.assertEqual(int_to_str_token(iter_val), token)

                # return 2 agents and 2 static objects
                for idx in range(0, 4, 1):
                    box = OrientedBox(center=StateSE2(x=10, y=10, heading=10), length=10, width=10, height=10)

                    metadata = SceneObjectMetadata(
                        token=int_to_str_token(idx + str_token_to_int(token)),
                        track_token=int_to_str_token(idx + str_token_to_int(token) + 100),
                        track_id=None,
                        timestamp_us=0,
                        category_name="foo",
                    )

                    if idx < 2:
                        yield Agent(
                            tracked_object_type=TrackedObjectType.VEHICLE,
                            oriented_box=box,
                            velocity=StateVector2D(x=10, y=10),
                            metadata=metadata,
                        )
                    else:
                        yield StaticObject(
                            tracked_object_type=TrackedObjectType.CZONE_SIGN, oriented_box=box, metadata=metadata
                        )

            # Mock the interpolation so that validating the data is easier
            def interpolate_future_waypoints_patch(
                waypoints: List[InterpolatableState], time_horizon: float, interval_s: float
            ) -> List[Optional[InterpolatableState]]:
                """
                The patch for interpolate_future_waypoints that validates the arguments and generates fake data.
                """
                self.assertEqual(4, len(waypoints))
                self.assertEqual(0.5, interval_s)
                self.assertEqual(5, time_horizon)

                return waypoints

            def future_waypoints_for_agents_patch(
                log_file: str, agents_tokens: List[str], start_time: int, end_time: int
            ) -> Generator[Tuple[str, Waypoint], None, None]:
                """
                The patch for get_future_waypoints_for_agents_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual("data_root/log_name.db", log_file)
                self.assertEqual(iter_val * 1e6, start_time)
                self.assertEqual((iter_val + 5.5) * 1e6, end_time)
                self.assertEqual(2, len(agents_tokens))

                # tokens can be provided in any order
                check_tokens = [str_token_to_int(t) for t in agents_tokens]
                check_tokens.sort()
                self.assertEqual(iter_val + 100, check_tokens[0])
                self.assertEqual(iter_val + 100 + 1, check_tokens[1])

                # generate fake data
                for i in range(8):
                    waypoint = Waypoint(
                        time_point=TimePoint(time_us=i),
                        oriented_box=OrientedBox(center=StateSE2(x=i, y=i, heading=i), length=i, width=i, height=i),
                        velocity=None,
                    )

                    token = check_tokens[0] if i < 4 else check_tokens[1]

                    yield (int_to_str_token(token), waypoint)

            with mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
                download_file_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db",
                lidarpc_tokens_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_for_lidarpc_token_from_db",
                tracked_objects_for_token_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db",
                future_waypoints_for_agents_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sensor_data_token_timestamp_from_db",
                get_token_timestamp_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.interpolate_future_waypoints",
                interpolate_future_waypoints_patch,
            ):
                scenario = self._make_test_scenario()
                agents = scenario.get_tracked_objects_at_iteration(iter_val, ground_truth_predictions)
                objects = agents.tracked_objects.tracked_objects
                self.assertEqual(4, len(objects))

                # Objects can be returned in any order, sort by track_token for ease of validation
                objects.sort(key=lambda x: str_token_to_int(x.metadata.token))

                # First two objects should be agents
                for i in range(0, 2, 1):
                    test_obj = objects[i]
                    self.assertTrue(isinstance(test_obj, Agent))
                    self.assertEqual(iter_val + i, str_token_to_int(test_obj.metadata.token))
                    self.assertEqual(iter_val + i + 100, str_token_to_int(test_obj.metadata.track_token))
                    self.assertEqual(TrackedObjectType.VEHICLE, test_obj.tracked_object_type)
                    self.assertIsNotNone(test_obj.predictions)
                    object_waypoints = test_obj.predictions[0].waypoints
                    self.assertEqual(4, len(object_waypoints))
                    for j in range(len(object_waypoints)):
                        self.assertEqual(j + (i * len(object_waypoints)), object_waypoints[j].x)

                # Next two objects should be static objects
                for i in range(2, 4, 1):
                    test_obj = objects[i]
                    self.assertTrue(isinstance(test_obj, StaticObject))
                    self.assertEqual(iter_val + i, str_token_to_int(test_obj.metadata.token))
                    self.assertEqual(iter_val + i + 100, str_token_to_int(test_obj.metadata.track_token))
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, test_obj.tracked_object_type)

    def test_get_tracked_objects_within_time_window_at_iteration(self) -> None:
        """
        Tests that the get_tracked_objects_within_time_window_at_iteration method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
            expected_sensor_data_source=get_lidarpc_sensor_data(),
            expected_start_timestamp=int((1 * 1e6) + 2345),
            expected_end_timestamp=int((21 * 1e6) + 2345),
            expected_subsample_step=2,
        )

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)

        for iter_val in [3, 4]:

            def get_token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch for get_sensor_data_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual("data_root/log_name.db", log_file)
                self.assertEqual(
                    SensorDataSource(
                        table='lidar_pc',
                        sensor_table='lidar',
                        sensor_token_column='lidar_token',
                        channel='MergedPointCloud',
                    ),
                    sensor_source,
                )
                self.assertEqual(int_to_str_token(iter_val), token)

                return int(iter_val * 1e6)

            def tracked_objects_within_time_interval_patch(
                log_file: str, start_timestamp: int, end_timestamp: int, filter_tokens: Optional[Set[str]]
            ) -> Generator[TrackedObject, None, None]:
                """
                The patch for get_tracked_objects_for_lidarpc_token that validates the arguments and generates fake data.
                """
                self.assertEqual("data_root/log_name.db", log_file)
                self.assertEqual((iter_val - 2) * 1e6, start_timestamp)
                self.assertEqual((iter_val + 2) * 1e6, end_timestamp)
                self.assertIsNone(filter_tokens)

                for time_idx in range(-2, 3, 1):
                    # return 2 agents and 2 static objects
                    for idx in range(0, 4, 1):
                        box = OrientedBox(center=StateSE2(x=10, y=10, heading=10), length=10, width=10, height=10)

                        metadata = SceneObjectMetadata(
                            token=int_to_str_token(idx + iter_val),
                            track_token=int_to_str_token(idx + iter_val + 100),
                            track_id=None,
                            timestamp_us=(iter_val + time_idx) * 1e6,
                            category_name="foo",
                        )

                        if idx < 2:
                            yield Agent(
                                tracked_object_type=TrackedObjectType.VEHICLE,
                                oriented_box=box,
                                velocity=StateVector2D(x=10, y=10),
                                metadata=metadata,
                            )
                        else:
                            yield StaticObject(
                                tracked_object_type=TrackedObjectType.CZONE_SIGN, oriented_box=box, metadata=metadata
                            )

            # Mock the interpolation so that validating the data is easier
            def interpolate_future_waypoints_patch(
                waypoints: List[InterpolatableState], time_horizon: float, interval_s: float
            ) -> List[Optional[InterpolatableState]]:
                """
                The patch for interpolate_future_waypoints that validates the arguments and generates fake data.
                """
                self.assertEqual(4, len(waypoints))
                self.assertEqual(0.5, interval_s)
                self.assertEqual(5, time_horizon)

                return waypoints

            def future_waypoints_for_agents_patch(
                log_file: str, agents_tokens: List[str], start_time: int, end_time: int
            ) -> Generator[Tuple[str, Waypoint], None, None]:
                """
                The patch for get_future_waypoints_for_agents_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual("data_root/log_name.db", log_file)
                self.assertEqual(end_time - start_time, 5.5 * 1e6)
                self.assertEqual(2, len(agents_tokens))

                # tokens can be provided in any order
                check_tokens = [str_token_to_int(t) for t in agents_tokens]
                check_tokens.sort()
                self.assertEqual(iter_val + 100, check_tokens[0])
                self.assertEqual(iter_val + 100 + 1, check_tokens[1])

                # generate fake data
                for i in range(8):
                    waypoint = Waypoint(
                        time_point=TimePoint(time_us=i),
                        oriented_box=OrientedBox(center=StateSE2(x=i, y=i, heading=i), length=i, width=i, height=i),
                        velocity=None,
                    )

                    token = check_tokens[0] if i < 4 else check_tokens[1]

                    yield (int_to_str_token(token), waypoint)

            with mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
                download_file_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db",
                lidarpc_tokens_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_within_time_interval_from_db",
                tracked_objects_within_time_interval_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db",
                future_waypoints_for_agents_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sensor_data_token_timestamp_from_db",
                get_token_timestamp_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.interpolate_future_waypoints",
                interpolate_future_waypoints_patch,
            ):
                scenario = self._make_test_scenario()
                agents = scenario.get_tracked_objects_within_time_window_at_iteration(
                    iter_val,
                    2,
                    2,
                    future_trajectory_sampling=ground_truth_predictions,
                )
                objects = agents.tracked_objects.tracked_objects
                self.assertEqual(20, len(objects))

                # For the given queries, there should be 5 windows of objects returned.
                # Objects are sorted by type, then by timestamp.
                # So, we should see [Agent1_1, Agent1_2, Agent2_1, Agent2_2, ..., SO1_1, SO_1_2, ...]
                #   for (type)(timestamp)_(track_token)
                num_objects = 2
                for window in range(0, 5, 1):
                    for object_num in range(0, 2, 1):
                        start_agent_idx = window * 2

                        # Agent
                        test_obj = objects[start_agent_idx + object_num]
                        self.assertTrue(isinstance(test_obj, Agent))
                        self.assertEqual(iter_val + object_num, str_token_to_int(test_obj.metadata.token))
                        self.assertEqual(iter_val + object_num + 100, str_token_to_int(test_obj.metadata.track_token))
                        self.assertEqual(TrackedObjectType.VEHICLE, test_obj.tracked_object_type)
                        self.assertIsNotNone(test_obj.predictions)
                        object_waypoints = test_obj.predictions[0].waypoints
                        self.assertEqual(4, len(object_waypoints))
                        for j in range(len(object_waypoints)):
                            self.assertEqual(j + (object_num * len(object_waypoints)), object_waypoints[j].x)

                        # Static object
                        start_obj_idx = 10 + (window * 2)
                        test_obj = objects[start_obj_idx + object_num]
                        self.assertTrue(isinstance(test_obj, StaticObject))
                        self.assertEqual(iter_val + object_num + num_objects, str_token_to_int(test_obj.metadata.token))
                        self.assertEqual(
                            iter_val + object_num + num_objects + 100, str_token_to_int(test_obj.metadata.track_token)
                        )
                        self.assertEqual(TrackedObjectType.CZONE_SIGN, test_obj.tracked_object_type)

    def test_nuplan_scenario_memory_usage(self) -> None:
        """
        Test that repeatedly creating and destroying nuplan scenario does not cause memory leaks.
        """
        starting_usage = 0
        ending_usage = 0
        num_iterations = 5

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
            download_file_patch_fxn,
        ):
            hpy = guppy.hpy()
            hpy.setrelheap()

            for i in range(0, num_iterations, 1):
                # Use nested function to ensure local handles go out of scope
                scenario = self._make_test_scenario()

                # Force the scenario to be used
                _ = scenario.token

                gc.collect()

                heap = hpy.heap()

                # Force heapy to materialize the heap statistics
                # This is done lazily, which can lead to noise if not forced.
                _ = heap.size

                # Skip the first few iterations - there can be noise as caches fill up
                if i == num_iterations - 2:
                    starting_usage = heap.size
                if i == num_iterations - 1:
                    ending_usage = heap.size

            memory_difference_in_mb = (ending_usage - starting_usage) / (1024 * 1024)

            # Alert on either 100 kb growth or 10 % of starting usage, whichever is bigger
            max_allowable_growth_mb = max(0.1, 0.1 * starting_usage / (1024 * 1024))
            self.assertGreater(max_allowable_growth_mb, memory_difference_in_mb)

    @patch(f"{TEST_PATH}.LocalStore", autospec=True)
    @patch(f"{TEST_PATH}.S3Store", autospec=True)
    @patch(f"{TEST_PATH}.os.getenv")
    def test_get_sensors_at_iteration(self, mock_get_env: Mock, mock_s3_store: Mock, mock_local_store: Mock) -> None:
        """Test get_sensors_at_iteration."""
        mock_url = "url"
        mock_get_env.side_effect = ["s3", mock_url]
        mock_s3_store.return_value = Mock(spec_set=S3Store)
        mock_local_store.return_value = Mock(spec_set=LocalStore)

        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
            expected_sensor_data_source=get_lidarpc_sensor_data(),
            expected_start_timestamp=int(1 * 1e6) + 2345,
            expected_end_timestamp=int(21 * 1e6) + 2345,
            expected_subsample_step=2,
        )

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )
        with mock.patch(f"{TEST_PATH}.download_file_if_necessary", download_file_patch_fxn):
            scenario = self._make_test_scenario()

        for iter_val in [0, 3, 5]:
            lidar_token = int_to_str_token(iter_val)
            get_sensor_data_from_sensor_data_tokens_from_db_fxn = (
                self._get_sensor_data_from_sensor_data_tokens_from_db_patch(
                    expected_log_file="data_root/log_name.db",
                    expected_sensor_data_source=get_lidarpc_sensor_data(),
                    expected_sensor_class=LidarPc,
                    expected_tokens=[lidar_token],
                )
            )

            get_images_from_lidar_tokens_fxn = self._get_images_from_lidar_tokens_patch(
                expected_log_file="data_root/log_name.db",
                expected_tokens=[lidar_token],
                expected_channels=[CameraChannel.CAM_R0.value, LidarChannel.MERGED_PC.value],
                expected_lookahead_window_us=50000,
                expected_lookback_window_us=50000,
            )

            load_lidar_fxn = self._load_point_cloud_patch(
                LidarPc(
                    token=lidar_token,
                    next_token=lidar_token,
                    prev_token=lidar_token,
                    ego_pose_token=lidar_token,
                    lidar_token=lidar_token,
                    scene_token=lidar_token,
                    filename=f"lidar_{lidar_token}",
                    timestamp=str_token_to_int(lidar_token),
                ),
                mock_local_store.return_value,
                mock_s3_store.return_value,
            )

            load_image_fxn = self._load_image_patch(mock_local_store.return_value, mock_s3_store.return_value)

            with mock.patch(
                f"{TEST_PATH_UTILS}.get_sampled_sensor_tokens_in_time_window_from_db",
                lidarpc_tokens_patch_fxn,
            ), mock.patch(
                f"{TEST_PATH}.get_sensor_data_from_sensor_data_tokens_from_db",
                get_sensor_data_from_sensor_data_tokens_from_db_fxn,
            ), mock.patch(
                f"{TEST_PATH}.get_images_from_lidar_tokens",
                get_images_from_lidar_tokens_fxn,
            ), mock.patch(
                f"{TEST_PATH}.load_point_cloud",
                load_lidar_fxn,
            ), mock.patch(
                f"{TEST_PATH}.load_image",
                load_image_fxn,
            ):
                sensors = scenario.get_sensors_at_iteration(iter_val, [CameraChannel.CAM_R0, LidarChannel.MERGED_PC])
                self.assertEqual(LidarChannel.MERGED_PC, list(sensors.pointcloud.keys())[0])
                self.assertEqual(CameraChannel.CAM_R0, list(sensors.images.keys())[0])
                mock_local_store.assert_called_with("sensor_root")
                mock_s3_store.assert_called_with(f"{mock_url}/sensor_blobs", show_progress=True)

    @patch(f"{TEST_PATH}.LocalStore", autospec=True)
    @patch(f"{TEST_PATH}.S3Store", autospec=True)
    @patch(f"{TEST_PATH}.os.getenv")
    def test_get_past_sensors(self, mock_get_env: Mock, mock_s3_store: Mock, mock_local_store: Mock) -> None:
        """Test get_past_sensors."""
        mock_url = "url"
        mock_get_env.side_effect = ["s3", mock_url]
        mock_s3_store.return_value = Mock(spec_set=S3Store)
        mock_local_store.return_value = Mock(spec_set=LocalStore)

        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
            expected_sensor_data_source=get_lidarpc_sensor_data(),
            expected_start_timestamp=int((1 * 1e6) + 2345),
            expected_end_timestamp=int((21 * 1e6) + 2345),
            expected_subsample_step=2,
        )
        lidar_token = int_to_str_token(9)

        get_sampled_lidarpcs_from_db_fxn = self._get_sampled_lidarpcs_from_db_patch(
            expected_log_file="data_root/log_name.db",
            expected_initial_token=int_to_str_token(0),
            expected_sensor_data_source=get_lidarpc_sensor_data(),
            expected_sample_indexes=[9],
            expected_future=False,
        )

        get_images_from_lidar_tokens_fxn = self._get_images_from_lidar_tokens_patch(
            expected_log_file="data_root/log_name.db",
            expected_tokens=[lidar_token],
            expected_channels=[CameraChannel.CAM_R0.value, LidarChannel.MERGED_PC.value],
            expected_lookahead_window_us=50000,
            expected_lookback_window_us=50000,
        )

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )
        load_lidar_fxn = self._load_point_cloud_patch(
            LidarPc(
                token=lidar_token,
                next_token=lidar_token,
                prev_token=lidar_token,
                ego_pose_token=lidar_token,
                lidar_token=lidar_token,
                scene_token=lidar_token,
                filename=f"lidar_{lidar_token}",
                timestamp=str_token_to_int(lidar_token),
            ),
            mock_local_store.return_value,
            mock_s3_store.return_value,
        )

        load_image_fxn = self._load_image_patch(mock_local_store.return_value, mock_s3_store.return_value)

        with mock.patch(f"{TEST_PATH}.download_file_if_necessary", download_file_patch_fxn,), mock.patch(
            f"{TEST_PATH_UTILS}.get_sampled_sensor_tokens_in_time_window_from_db",
            lidarpc_tokens_patch_fxn,
        ), mock.patch(f"{TEST_PATH}.get_sampled_lidarpcs_from_db", get_sampled_lidarpcs_from_db_fxn,), mock.patch(
            f"{TEST_PATH}.get_images_from_lidar_tokens",
            get_images_from_lidar_tokens_fxn,
        ), mock.patch(
            f"{TEST_PATH}.load_point_cloud",
            load_lidar_fxn,
        ), mock.patch(
            f"{TEST_PATH}.load_image",
            load_image_fxn,
        ):
            scenario = self._make_test_scenario()
            past_sensors = list(
                scenario.get_past_sensors(
                    iteration=0,
                    time_horizon=0.4,
                    num_samples=1,
                    channels=[CameraChannel.CAM_R0, LidarChannel.MERGED_PC],
                )
            )
            self.assertEqual(1, len(past_sensors))
            self.assertEqual(LidarChannel.MERGED_PC, list(past_sensors[0].pointcloud.keys())[0])
            self.assertEqual(CameraChannel.CAM_R0, list(past_sensors[0].images.keys())[0])
            mock_local_store.assert_called_with("sensor_root")
            mock_s3_store.assert_called_with(f"{mock_url}/sensor_blobs", show_progress=True)

    @patch(f"{TEST_PATH}.download_file_if_necessary", Mock())
    @patch(f"{TEST_PATH}.absolute_path_to_log_name", Mock())
    @patch(f"{TEST_PATH}.get_images_from_lidar_tokens", Mock(return_value=[]))
    @patch(f"{TEST_PATH}.NuPlanScenario._find_matching_lidar_pcs")
    @patch(f"{TEST_PATH}.load_point_cloud")
    @patch(f"{TEST_PATH}.load_image")
    def test_get_past_sensors_no_channels(
        self, mock_load_image: Mock, mock_load_point_cloud: Mock, mock__find_matching_lidar_pcs: Mock
    ) -> None:
        """Test get_past_sensors when no channels are passed."""
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.token = "token"
        mock_load_point_cloud.return_value = Mock(spec_set=LidarPointCloud)
        mock__find_matching_lidar_pcs.return_value = iter([mock_lidar_pc])
        scenario = self._make_test_scenario()
        past_sensors = list(
            scenario.get_past_sensors(
                iteration=0,
                time_horizon=0.4,
                num_samples=1,
                channels=None,
            )
        )
        mock__find_matching_lidar_pcs.assert_called_once()
        mock_load_point_cloud.assert_called_once()

        # Explicitly check that loading images is not
        mock_load_image.assert_not_called()
        self.assertIsNone(past_sensors[0].images)
        self.assertIsNotNone(past_sensors[0].pointcloud)

    @patch(f"{TEST_PATH}.download_file_if_necessary", Mock())
    @patch(f"{TEST_PATH}.absolute_path_to_log_name", Mock())
    @patch(f"{TEST_PATH}.get_images_from_lidar_tokens", Mock(return_value=[]))
    @patch(f"{TEST_PATH}.extract_sensor_tokens_as_scenario", Mock(return_value=[None]))
    @patch(f"{TEST_PATH}.get_sensor_data_from_sensor_data_tokens_from_db")
    @patch(f"{TEST_PATH}.load_point_cloud")
    @patch(f"{TEST_PATH}.load_image")
    def test_get_sensors_at_iteration_no_channels(
        self,
        mock_load_image: Mock,
        mock_load_point_cloud: Mock,
        mock_get_sensor_data_from_sensor_data_tokens_from_db: Mock,
    ) -> None:
        """Test get_past_sensors when no channels are passed."""
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.token = "token"
        mock_load_point_cloud.return_value = Mock(spec_set=LidarPointCloud)
        mock_get_sensor_data_from_sensor_data_tokens_from_db.return_value = iter([mock_lidar_pc])
        scenario = self._make_test_scenario()
        sensors = scenario.get_sensors_at_iteration(
            iteration=0,
            channels=None,
        )
        mock_get_sensor_data_from_sensor_data_tokens_from_db.assert_called_once()
        mock_load_point_cloud.assert_called_once()

        # Explicitly check that loading images is not
        mock_load_image.assert_not_called()
        self.assertIsNone(sensors.images)
        self.assertIsNotNone(sensors.pointcloud)


if __name__ == '__main__':
    unittest.main()
