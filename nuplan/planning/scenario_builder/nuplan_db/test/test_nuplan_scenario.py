import gc
import unittest
from typing import Callable, Generator, List, Optional, Set, Tuple

import guppy
import mock

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
from nuplan.database.nuplan_db.test.minimal_db_test_utils import int_to_str_token, str_token_to_int
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


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
        )

    def _get_sampled_lidarpc_tokens_in_time_window_patch(
        self,
        expected_log_file: str,
        expected_start_timestamp: int,
        expected_end_timestamp: int,
        expected_subsample_step: int,
    ) -> Callable[[str, int, int, int], Generator[str, None, None]]:
        """
        Creates a patch for the get_sampled_lidarpc_tokens_in_time_window function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_start_timestamp: The expected start timestamp with which the function is expected to be called.
        :param expected_end_timestamp: The expected end timestamp with which the function is expected to be called.
        :param expected_subsample_step: The expected subsample step with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(
            actual_log_file: str, actual_start_timestamp: int, actual_end_timestamp: int, actual_subsample_step: int
        ) -> Generator[str, None, None]:
            """
            The patch function for get_sampled_lidarpc_tokens_in_time_window.
            """
            self.assertEqual(expected_log_file, actual_log_file)
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
        lidarpc_tokens_patch_fxn = self._get_sampled_lidarpc_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
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
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_lidarpc_tokens_in_time_window_from_db",
                lidarpc_tokens_patch_fxn,
            ):
                scenario = self._make_test_scenario()
                self.assertEqual(10, scenario.get_number_of_iterations())

    def test_get_time_point(self) -> None:
        """
        Tests that the get_time_point method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_lidarpc_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
            expected_start_timestamp=int((1 * 1e6) + 2345),
            expected_end_timestamp=int((21 * 1e6) + 2345),
            expected_subsample_step=2,
        )

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )

        for iter_val in [0, 3, 5]:

            def token_timestamp_patch(log_file: str, token: str) -> int:
                """
                The patch method for get_lidarpc_token_timstamp_from_db that validates the arguments.
                """
                self.assertEqual("data_root/log_name.db", log_file)
                self.assertEqual(int_to_str_token(iter_val), token)

                return int(str_token_to_int(iter_val) + 5)

            with mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary",
                download_file_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_lidarpc_tokens_in_time_window_from_db",
                lidarpc_tokens_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.get_lidarpc_token_timestamp_from_db",
                token_timestamp_patch,
            ):
                scenario = self._make_test_scenario()
                self.assertEqual(iter_val + 5, scenario.get_time_point(iter_val).time_us)

    def test_get_tracked_objects_at_iteration(self) -> None:
        """
        Tests that the get_tracked_objects_at_iteration method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_lidarpc_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
            expected_start_timestamp=int((1 * 1e6) + 2345),
            expected_end_timestamp=int((21 * 1e6) + 2345),
            expected_subsample_step=2,
        )

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)

        for iter_val in [0, 2, 3]:

            def get_token_timestamp_patch(log_file: str, token: str) -> int:
                """
                The patch for get_lidarpc_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual("data_root/log_name.db", log_file)
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
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_lidarpc_tokens_in_time_window_from_db",
                lidarpc_tokens_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_for_lidarpc_token_from_db",
                tracked_objects_for_token_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db",
                future_waypoints_for_agents_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_lidarpc_token_timestamp_from_db",
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
        lidarpc_tokens_patch_fxn = self._get_sampled_lidarpc_tokens_in_time_window_patch(
            expected_log_file="data_root/log_name.db",
            expected_start_timestamp=int((1 * 1e6) + 2345),
            expected_end_timestamp=int((21 * 1e6) + 2345),
            expected_subsample_step=2,
        )

        download_file_patch_fxn = self._get_download_file_if_necessary_patch(
            expected_data_root="data_root/", expected_log_file_load_path="data_root/log_name.db"
        )
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)

        for iter_val in [3, 4]:

            def get_token_timestamp_patch(log_file: str, token: str) -> int:
                """
                The patch for get_lidarpc_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual("data_root/log_name.db", log_file)
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
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_lidarpc_tokens_in_time_window_from_db",
                lidarpc_tokens_patch_fxn,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_within_time_interval_from_db",
                tracked_objects_within_time_interval_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db",
                future_waypoints_for_agents_patch,
            ), mock.patch(
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_lidarpc_token_timestamp_from_db",
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


if __name__ == '__main__':
    unittest.main()
