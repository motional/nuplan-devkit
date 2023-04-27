import math
import os
import unittest
from pathlib import Path
from typing import Dict, List, Tuple, cast

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.database.nuplan_db.image import Image
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplan_db_utils import SensorDataSource
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_cameras,
    get_ego_state_for_lidarpc_token_from_db,
    get_end_sensor_time_from_db,
    get_future_waypoints_for_agents_from_db,
    get_images_from_lidar_tokens,
    get_lidarpc_tokens_with_scenario_tag_from_db,
    get_mission_goal_for_sensor_data_token_from_db,
    get_roadblock_ids_for_lidarpc_token_from_db,
    get_sampled_ego_states_from_db,
    get_sampled_lidarpcs_from_db,
    get_sampled_sensor_tokens_in_time_window_from_db,
    get_scenarios_from_db,
    get_sensor_data_from_sensor_data_tokens_from_db,
    get_sensor_data_token_timestamp_from_db,
    get_sensor_token,
    get_sensor_token_by_index_from_db,
    get_sensor_token_map_name_from_db,
    get_sensor_transform_matrix_for_sensor_data_token_from_db,
    get_statese2_for_lidarpc_token_from_db,
    get_tracked_objects_for_lidarpc_token_from_db,
    get_tracked_objects_within_time_interval_from_db,
    get_traffic_light_status_for_lidarpc_token_from_db,
)
from nuplan.database.nuplan_db.test.minimal_db_test_utils import (
    DBGenerationParameters,
    generate_minimal_nuplan_db,
    int_to_str_token,
    str_token_to_int,
)


class TestNuPlanScenarioQueries(unittest.TestCase):
    """
    Test suite for the NuPlan scenario queries.
    """

    generation_parameters: DBGenerationParameters

    @staticmethod
    def getDBFilePath() -> Path:
        """
        Get the location for the temporary SQLite file used for the test DB.
        :return: The filepath for the test data.
        """
        return Path("/tmp/test_nuplan_scenario_queries.sqlite3")

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create the mock DB data.
        """
        db_file_path = TestNuPlanScenarioQueries.getDBFilePath()
        if db_file_path.exists():
            db_file_path.unlink()

        cls.generation_parameters = DBGenerationParameters(
            num_lidars=1,
            num_cameras=2,
            num_sensor_data_per_sensor=50,
            num_lidarpc_per_image_ratio=2,
            num_scenes=10,
            num_traffic_lights_per_lidar_pc=5,
            num_agents_per_lidar_pc=3,
            num_static_objects_per_lidar_pc=2,
            scene_scenario_tag_mapping={
                5: ["first_tag"],
                6: ["first_tag", "second_tag"],
                7: ["second_tag"],
            },
            file_path=str(db_file_path),
        )

        generate_minimal_nuplan_db(cls.generation_parameters)

    def setUp(self) -> None:
        """
        The method to run before each test.
        """
        self.db_file_name = str(TestNuPlanScenarioQueries.getDBFilePath())
        self.sensor_source = SensorDataSource("lidar_pc", "lidar", "lidar_token", "channel")

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Destroy the mock DB data.
        """
        db_file_path = TestNuPlanScenarioQueries.getDBFilePath()
        if os.path.exists(db_file_path):
            os.remove(db_file_path)

    def test_get_sensor_token_from_index(self) -> None:
        """
        Test the get_sensor_token_from_index query.
        """
        for sample_index in [0, 12, 24]:
            retrieved_token = get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, sample_index)

            self.assertEqual(sample_index / self.generation_parameters.num_lidars, str_token_to_int(retrieved_token))

        # Check that a index out of range returns None
        self.assertIsNone(get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, 100000))

        # Negative indexes should throw.
        with self.assertRaises(ValueError):
            get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, -2)

    def test_get_end_sensor_time_from_db(self) -> None:
        """
        Test the get_end_sensor_time_from_db query.
        """
        log_end_time = get_end_sensor_time_from_db(self.db_file_name, sensor_source=self.sensor_source)
        self.assertEqual(49 * 1e6, log_end_time)

    def test_get_sensor_token_timestamp_from_db(self) -> None:
        """
        Test the get_sensor_data_token_timestamp_from_db query.
        """
        for token in [0, 3, 7]:
            expected_timestamp = token * 1e6
            actual_timestamp = get_sensor_data_token_timestamp_from_db(
                self.db_file_name, self.sensor_source, int_to_str_token(token)
            )
            self.assertEqual(expected_timestamp, actual_timestamp)

        # When token doesn't exist, function should return None
        self.assertIsNone(
            get_sensor_data_token_timestamp_from_db(self.db_file_name, self.sensor_source, int_to_str_token(1000))
        )

    def test_get_sensor_token_map_name_from_db(self) -> None:
        """
        Test the get_sensor_token_map_name_from_db query.
        """
        for token in [0, 2, 6]:
            expected_map_name = "map_version"
            actual_map_name = get_sensor_token_map_name_from_db(
                self.db_file_name,
                self.sensor_source,
                int_to_str_token(token),
            )

            self.assertEqual(expected_map_name, actual_map_name)

        self.assertIsNone(
            get_sensor_token_map_name_from_db(
                self.db_file_name,
                self.sensor_source,
                int_to_str_token(1000),
            )
        )

    def test_get_sampled_sensor_tokens_in_time_window_from_db(self) -> None:
        """
        Test the get_sampled_lidarpc_tokens_in_time_window_from_db query.
        """
        expected_tokens = [10, 13, 16, 19]
        actual_tokens = list(
            str_token_to_int(v)
            for v in get_sampled_sensor_tokens_in_time_window_from_db(
                log_file=self.db_file_name,
                sensor_source=self.sensor_source,
                start_timestamp=int(10 * 1e6),
                end_timestamp=int(20 * 1e6),
                subsample_interval=3,
            )
        )

        self.assertEqual(expected_tokens, actual_tokens)

    def test_get_sensor_data_from_sensor_data_tokens_from_db(self) -> None:
        """
        Test the get_sensor_data_from_sensor_data_tokens_from_db query.
        """
        lidar_pc_tokens = [int_to_str_token(v) for v in [10, 13, 21]]
        image_tokens = [int_to_str_token(v) for v in [1100000]]

        lidar_pcs = [
            cast(LidarPc, sensor_data)
            for sensor_data in get_sensor_data_from_sensor_data_tokens_from_db(
                self.db_file_name, self.sensor_source, LidarPc, lidar_pc_tokens
            )
        ]
        images = [
            cast(Image, sensor_data)
            for sensor_data in get_sensor_data_from_sensor_data_tokens_from_db(
                self.db_file_name, SensorDataSource("image", "camera", "camera_token", "camera_0"), Image, image_tokens
            )
        ]

        self.assertEqual(len(lidar_pc_tokens), len(lidar_pcs))
        self.assertEqual(len(image_tokens), len(images))

        # Results are not necessarily sorted, so sort for the unit test.
        lidar_pcs.sort(key=lambda x: int(x.timestamp))

        self.assertEqual(10, str_token_to_int(lidar_pcs[0].token))
        self.assertEqual(13, str_token_to_int(lidar_pcs[1].token))
        self.assertEqual(21, str_token_to_int(lidar_pcs[2].token))

        self.assertEqual(1100000, str_token_to_int(images[0].token))

    def test_get_lidar_transform_matrix_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_sensor_transform_matrix_for_sensor_data_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            xform_mat = get_sensor_transform_matrix_for_sensor_data_token_from_db(
                self.db_file_name,
                self.sensor_source,
                int_to_str_token(sample_token),
            )

            self.assertIsNotNone(xform_mat)
            self.assertEqual(xform_mat[0, 3], 0)

    def test_get_mission_goal_for_sensor_data_token_from_db(self) -> None:
        """
        Test the get_mission_goal_for_sensor_data_token_from_db query.
        """
        # With 10 scenes for 50 lidar_pcs, 12 will fall into scene
        #   10-14
        query_lidarpc_token = int_to_str_token(12)

        expected_ego_pose_x = 14
        expected_ego_pose_y = 15

        result = get_mission_goal_for_sensor_data_token_from_db(
            self.db_file_name, self.sensor_source, query_lidarpc_token
        )

        self.assertIsNotNone(result)
        self.assertEqual(expected_ego_pose_x, result.x)
        self.assertEqual(expected_ego_pose_y, result.y)

    def test_get_roadblock_ids_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_roadblock_ids_for_lidarpc_token_from_db query.
        """
        # roadblock ids are in the scene table as a space-separated list of route roadblock IDS, e.g. "123 234 345"
        result = get_roadblock_ids_for_lidarpc_token_from_db(self.db_file_name, int_to_str_token(0))
        self.assertEqual(result, ['0', '1', '2'])

    def test_get_statese2_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_statese2_for_lidarpc_token_from_db query.
        """
        query_lidarpc_token = int_to_str_token(13)

        expected_ego_pose_x = 13
        expected_ego_pose_y = 14

        result = get_statese2_for_lidarpc_token_from_db(self.db_file_name, query_lidarpc_token)

        self.assertIsNotNone(result)
        self.assertEqual(expected_ego_pose_x, result.x)
        self.assertEqual(expected_ego_pose_y, result.y)

    def test_get_sampled_lidarpcs_from_db(self) -> None:
        """
        Test the get_sampled_lidarpcs_from_db query.
        """
        test_cases = [
            {"initial_token": 5, "sample_indexes": [0, 1, 2], "future": True, "expected_return_tokens": [5, 6, 7]},
            {"initial_token": 5, "sample_indexes": [0, 1, 2], "future": False, "expected_return_tokens": [3, 4, 5]},
            {"initial_token": 7, "sample_indexes": [0, 3, 12], "future": False, "expected_return_tokens": [4, 7]},
            {"initial_token": 0, "sample_indexes": [1000], "future": True, "expected_return_tokens": []},
        ]

        for test_case in test_cases:
            initial_token = int_to_str_token(test_case["initial_token"])
            expected_return_tokens = [int_to_str_token(v) for v in test_case["expected_return_tokens"]]  # type: ignore

            actual_returned_lidarpcs = list(
                get_sampled_lidarpcs_from_db(
                    self.db_file_name,
                    initial_token,
                    self.sensor_source,
                    test_case["sample_indexes"],
                    test_case["future"],
                )
            )

            self.assertEqual(len(expected_return_tokens), len(actual_returned_lidarpcs))
            for i in range(len(expected_return_tokens)):
                self.assertEqual(expected_return_tokens[i], actual_returned_lidarpcs[i].token)

    def test_get_sampled_ego_states_from_db(self) -> None:
        """
        Test the get_sampled_ego_states_from_db query.
        """
        test_cases = [
            {"initial_token": 5, "sample_indexes": [0, 1, 2], "future": True, "expected_row_indexes": [5, 6, 7]},
            {"initial_token": 5, "sample_indexes": [0, 1, 2], "future": False, "expected_row_indexes": [3, 4, 5]},
            {"initial_token": 7, "sample_indexes": [0, 3, 12], "future": False, "expected_row_indexes": [4, 7]},
            {"initial_token": 0, "sample_indexes": [1000], "future": True, "expected_row_indexes": []},
        ]

        for test_case in test_cases:
            initial_token = int_to_str_token(test_case["initial_token"])
            expected_row_indexes = test_case["expected_row_indexes"]

            actual_returned_ego_states = list(
                get_sampled_ego_states_from_db(
                    self.db_file_name,
                    initial_token,
                    self.sensor_source,
                    test_case["sample_indexes"],
                    test_case["future"],
                )
            )

            self.assertEqual(len(expected_row_indexes), len(actual_returned_ego_states))  # type: ignore
            for i in range(len(expected_row_indexes)):  # type: ignore
                self.assertEqual(
                    expected_row_indexes[i] * 1e6, actual_returned_ego_states[i].time_point.time_us  # type: ignore
                )

    def test_get_ego_state_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_ego_state_for_lidarpc_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            returned_pose = get_ego_state_for_lidarpc_token_from_db(self.db_file_name, query_token)

            self.assertEqual(sample_token * 1e6, returned_pose.time_point.time_us)

    def test_get_traffic_light_status_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_traffic_light_status_for_lidarpc_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            traffic_light_statuses = list(
                get_traffic_light_status_for_lidarpc_token_from_db(self.db_file_name, query_token)
            )

            self.assertEqual(5, len(traffic_light_statuses))
            for tl_status in traffic_light_statuses:
                self.assertEqual(sample_token * 1e6, tl_status.timestamp)

    def test_get_tracked_objects_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_tracked_objects_for_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            tracked_objects = list(get_tracked_objects_for_lidarpc_token_from_db(self.db_file_name, query_token))

            self.assertEqual(5, len(tracked_objects))

            agent_count = 0
            static_object_count = 0

            # Some DB constants
            track_token_base_id = 600000  # The starting point for the track tokens
            token_base_id = 500000  # The starting point for the tokens
            token_sample_step = 10000  # The amount the tokens increment from the base for each lidar_pc iteration

            for idx, tracked_object in enumerate(tracked_objects):
                expected_track_token = track_token_base_id + idx
                expected_token = token_base_id + (token_sample_step * sample_token) + idx

                self.assertEqual(int_to_str_token(expected_track_token), tracked_object.track_token)
                self.assertEqual(int_to_str_token(expected_token), tracked_object.token)

                if isinstance(tracked_object, Agent):
                    agent_count += 1
                    self.assertEqual(TrackedObjectType.VEHICLE, tracked_object.tracked_object_type)
                    self.assertEqual(0, len(tracked_object.predictions))
                elif isinstance(tracked_object, StaticObject):
                    static_object_count += 1
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, tracked_object.tracked_object_type)
                else:
                    raise ValueError(f"Unexpected type: {type(tracked_object)}")

            self.assertEqual(3, agent_count)
            self.assertEqual(2, static_object_count)

    def test_get_tracked_objects_within_time_interval_from_db(self) -> None:
        """
        Test the get_tracked_objects_within_time_interval_from_db query.
        """
        # For this test, each "window" has 5 agents. 3 of which are agents, 2 generic objects.
        # This dict tests the edge conditions - e.g. for the first token, there aren't any past windows to sample.
        expected_num_windows = {0: 3, 30: 5, 48: 4}

        # An index used for helping with the math to check for the number of previous windows.
        expected_backward_offset = {0: 0, 30: -2, 48: -2}

        for sample_token in expected_num_windows.keys():
            start_timestamp = int(1e6 * (sample_token - 2))
            end_timestamp = int(1e6 * (sample_token + 2))

            tracked_objects = list(
                get_tracked_objects_within_time_interval_from_db(
                    self.db_file_name, start_timestamp, end_timestamp, filter_track_tokens=None
                )
            )

            expected_num_tokens = expected_num_windows[sample_token] * 5
            self.assertEqual(expected_num_tokens, len(tracked_objects))

            agent_count = 0
            static_object_count = 0

            # Some DB constants
            track_token_base_id = 600000  # The starting point for the track tokens
            token_base_id = 500000  # The starting point for the tokens
            token_sample_step = 10000  # The amount the tokens increment from the base for each lidar_pc iteration

            for idx, tracked_object in enumerate(tracked_objects):
                expected_track_token = track_token_base_id + (idx % 5)
                expected_token = (
                    token_base_id
                    + (
                        token_sample_step
                        * ((sample_token + expected_backward_offset[sample_token]) + math.floor(idx / 5))
                    )
                    + (idx % 5)
                )

                self.assertEqual(int_to_str_token(expected_track_token), tracked_object.track_token)
                self.assertEqual(int_to_str_token(expected_token), tracked_object.token)

                if isinstance(tracked_object, Agent):
                    agent_count += 1
                    self.assertEqual(TrackedObjectType.VEHICLE, tracked_object.tracked_object_type)
                    self.assertEqual(0, len(tracked_object.predictions))
                elif isinstance(tracked_object, StaticObject):
                    static_object_count += 1
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, tracked_object.tracked_object_type)
                else:
                    raise ValueError(f"Unexpected type: {type(tracked_object)}")

            self.assertEqual(3 * expected_num_windows[sample_token], agent_count)
            self.assertEqual(2 * expected_num_windows[sample_token], static_object_count)

    def test_get_future_waypoints_for_agents_from_db(self) -> None:
        """
        Test the get_future_waypoints_for_agents_from_db query.
        """
        track_tokens = [600000, 600001, 600002]
        start_timestamp = 0
        end_timestamp = int((20 * 1e6) - 1)

        query_output: Dict[str, List[Waypoint]] = {}
        for token, waypoint in get_future_waypoints_for_agents_from_db(
            self.db_file_name, (int_to_str_token(t) for t in track_tokens), start_timestamp, end_timestamp
        ):
            if token not in query_output:
                query_output[token] = []
            query_output[token].append(waypoint)

        expected_keys = ["{:08d}".format(t) for t in track_tokens]
        self.assertEqual(len(expected_keys), len(query_output))
        for expected_key in expected_keys:
            self.assertTrue(expected_key in query_output)

            collected_waypoints = query_output[expected_key]
            self.assertEqual(20, len(collected_waypoints))

            for i in range(0, len(collected_waypoints), 1):
                self.assertEqual(i * 1e6, collected_waypoints[i].time_point.time_us)

    def test_get_scenarios_from_db(self) -> None:
        """
        Test the get_scenarios_from_db_query.
        """
        # No filters.
        no_filter_output: List[int] = []
        for row in get_scenarios_from_db(
            self.db_file_name,
            filter_tokens=None,
            filter_types=None,
            filter_map_names=None,
            include_invalid_mission_goals=False,
            include_cameras=False,
        ):
            no_filter_output.append(str_token_to_int(row["token"].hex()))

        self.assertEqual(list(range(10, 40, 1)), no_filter_output)

        # Token filters
        filter_tokens = [int_to_str_token(v) for v in [15, 30]]
        tokens_filter_output: List[int] = []
        for row in get_scenarios_from_db(
            self.db_file_name,
            filter_tokens=filter_tokens,
            filter_types=None,
            filter_map_names=None,
            include_invalid_mission_goals=False,
            include_cameras=False,
        ):
            tokens_filter_output.append(row["token"].hex())

        self.assertEqual(filter_tokens, tokens_filter_output)

        # Scenario filters
        filter_scenarios = ["first_tag"]
        extracted_rows: List[Tuple[int, str]] = []
        for row in get_scenarios_from_db(
            self.db_file_name,
            filter_tokens=None,
            filter_types=filter_scenarios,
            filter_map_names=None,
            include_invalid_mission_goals=False,
            include_cameras=False,
        ):
            extracted_rows.append((str_token_to_int(row["token"].hex()), row["scenario_type"]))

        self.assertEqual(2, len(extracted_rows))
        self.assertEqual(25, extracted_rows[0][0])
        self.assertEqual("first_tag", extracted_rows[0][1])
        self.assertEqual(30, extracted_rows[1][0])
        self.assertEqual("first_tag", extracted_rows[1][1])

        filter_scenarios = ["second_tag"]
        extracted_rows = []
        for row in get_scenarios_from_db(
            self.db_file_name,
            filter_tokens=None,
            filter_types=filter_scenarios,
            filter_map_names=None,
            include_invalid_mission_goals=False,
            include_cameras=False,
        ):
            extracted_rows.append((str_token_to_int(row["token"].hex()), row["scenario_type"]))

        self.assertEqual(2, len(extracted_rows))
        self.assertEqual(30, extracted_rows[0][0])
        self.assertEqual("second_tag", extracted_rows[0][1])
        self.assertEqual(35, extracted_rows[1][0])
        self.assertEqual("second_tag", extracted_rows[1][1])

        # Map filters
        filter_maps = ["map_version"]
        row_cnt = sum(
            1
            for _ in get_scenarios_from_db(
                self.db_file_name,
                filter_tokens=None,
                filter_types=None,
                filter_map_names=filter_maps,
                include_invalid_mission_goals=False,
                include_cameras=False,
            )
        )

        self.assertLess(0, row_cnt)

        filter_maps = ["map_that_does_not_exist"]
        row_cnt = sum(
            1
            for _ in get_scenarios_from_db(
                self.db_file_name,
                filter_tokens=None,
                filter_types=None,
                filter_map_names=filter_maps,
                include_invalid_mission_goals=False,
                include_cameras=False,
            )
        )

        self.assertEqual(0, row_cnt)

        # Filter for camera token
        row_cnt = sum(
            1
            for _ in get_scenarios_from_db(
                self.db_file_name,
                filter_tokens=None,
                filter_types=None,
                filter_map_names=None,
                include_invalid_mission_goals=False,
                include_cameras=True,
            )
        )

        self.assertEqual(15, row_cnt)

        # All filters
        row_cnt = sum(
            1
            for _ in get_scenarios_from_db(
                self.db_file_name,
                filter_tokens=[int_to_str_token(25)],
                filter_types=["first_tag"],
                filter_map_names=["map_version"],
                include_invalid_mission_goals=False,
                include_cameras=False,
            )
        )

        self.assertEqual(1, row_cnt)

    def test_get_lidarpc_tokens_with_scenario_tag_from_db(self) -> None:
        """
        Test the get_lidarpc_tokens_with_scenario_tag_from_db query.
        """
        tuples = list(get_lidarpc_tokens_with_scenario_tag_from_db(self.db_file_name))
        self.assertEqual(4, len(tuples))

        expected_tuples = [
            ("first_tag", int_to_str_token(25)),
            ("first_tag", int_to_str_token(30)),
            ("second_tag", int_to_str_token(30)),
            ("second_tag", int_to_str_token(35)),
        ]

        for tup in tuples:
            self.assertTrue(tup in expected_tuples)

    def test_get_sensor_token(self) -> None:
        """Test the get_lidarpc_token_from_index query."""
        retrieved_token = get_sensor_token(self.db_file_name, 'lidar', 'channel')

        self.assertEqual(700000, str_token_to_int(retrieved_token))

        # Check that a missing channel  raises
        with self.assertRaisesRegex(RuntimeError, "Channel missing_channel not found in table lidar!"):
            self.assertIsNone(get_sensor_token(self.db_file_name, 'lidar', 'missing_channel'))

    def test_get_images_from_lidar_tokens(self) -> None:
        """Test the get_images_from_lidar_tokens query."""
        token = int_to_str_token(20)
        retrieved_images = list(
            get_images_from_lidar_tokens(self.db_file_name, [token], ["camera_0", "camera_1"], 50000, 50000)
        )

        self.assertEqual(2, len(retrieved_images))
        self.assertEqual(1100020, str_token_to_int(retrieved_images[0].token))
        self.assertEqual(1100070, str_token_to_int(retrieved_images[1].token))
        self.assertEqual("camera_0", retrieved_images[0].channel)
        self.assertEqual("camera_1", retrieved_images[1].channel)

    def test_get_cameras(self) -> None:
        """Test the get_cameras query."""
        retrieved_cameras = list(get_cameras(self.db_file_name, ["camera_0", "camera_1"]))

        self.assertEqual(2, len(retrieved_cameras))
        self.assertEqual(1000000, str_token_to_int(retrieved_cameras[0].token))
        self.assertEqual(1000001, str_token_to_int(retrieved_cameras[1].token))
        self.assertEqual("camera_0", retrieved_cameras[0].channel)
        self.assertEqual("camera_1", retrieved_cameras[1].channel)

        retrieved_cameras = list(get_cameras(self.db_file_name, ["camera_1"]))

        self.assertEqual(1, len(retrieved_cameras))
        self.assertEqual(1000001, str_token_to_int(retrieved_cameras[0].token))
        self.assertEqual("camera_1", retrieved_cameras[0].channel)


if __name__ == "__main__":
    unittest.main()
