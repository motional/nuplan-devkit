import unittest
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.tests.nuplan_db_test_utils import get_test_nuplan_db, get_test_nuplan_lidarpc
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class TestNuPlanScenario(unittest.TestCase):
    """
    Tests scenario for NuPlan
    """

    def setUp(self) -> None:
        """
        Initializes the hydra config
        """
        self.db = get_test_nuplan_db()
        self.lidarpc = get_test_nuplan_lidarpc()

        self.database_interval = 0.05  # 20Hz database
        scenario_duration = 20.0  # 20s scenario duration
        max_iterations = scenario_duration / self.database_interval  # 20s scenario at 20Hz
        subsample_ratio = 0.5  # scenario sub-sample ratio
        self.number_of_iterations = int(subsample_ratio * max_iterations)

        # This extraction token was experimentally figured out
        self.scenario = NuPlanScenario(
            db=self.db,
            log_name=self.lidarpc.log.logfile,
            initial_lidar_token=self.lidarpc.token,
            scenario_extraction_info=ScenarioExtractionInfo(
                DEFAULT_SCENARIO_NAME, scenario_duration, 0.0, subsample_ratio
            ),
            scenario_type=DEFAULT_SCENARIO_NAME,
            ego_vehicle_parameters=get_pacifica_parameters(),
            ground_truth_predictions=TrajectorySampling(time_horizon=8, num_poses=16),
        )

    def test_scenario(self) -> None:
        """
        Test reading tracked objects predictions from db
        """
        tracked_objects = self.scenario.get_tracked_objects_at_iteration(0).tracked_objects
        self.assertGreater(len(tracked_objects), 0)
        for agent in tracked_objects.get_agents():
            self.assertIsNotNone(agent.predictions)
            if agent.token in [
                "01937e3655c05e07",
                "538c5e990298506f",
                "b5f4cba46b6e5651",
                "e81f876abbf4576c",
                "b74095b495e857e6",
            ]:
                continue
            self.assertGreater(len(agent.predictions), 0)
            for prediction in agent.predictions:
                self.assertGreater(len(prediction.waypoints), 0)

    def test_properties(self) -> None:
        """
        Test properties of NuPlanScenario
        """
        scenario = self.scenario
        self.assertEqual(scenario.token, self.lidarpc.token)
        self.assertEqual(scenario.log_name, self.lidarpc.log.logfile)
        self.assertEqual(scenario.scenario_name, self.lidarpc.token)
        self.assertEqual(scenario.map_api.map_name, Path(self.lidarpc.log.map_version).stem)
        self.assertEqual(scenario.database_interval, self.database_interval)
        self.assertGreaterEqual(scenario.get_mission_goal().x, 0)
        self.assertGreaterEqual(scenario.get_mission_goal().y, 0)
        self.assertEqual(scenario.get_number_of_iterations(), self.number_of_iterations)

    def test_starting_from_middle_of_scenario(self) -> None:
        """
        Test that we can query future states from middle of trajectory
        """
        iteration = 50
        time_horizon = 1

        token = self.scenario._lidarpc_tokens[iteration]

        # Make sure that the future timestamps are actually correct
        lidar_pc = self.db.lidar_pc[token]
        timestamps = self.scenario.get_future_timestamps(iteration=iteration, time_horizon=time_horizon)
        for i in range(len(timestamps)):
            lidar_pc = lidar_pc.next
            self.assertEqual(lidar_pc.ego_pose.timestamp, timestamps[i].time_us)

        # Make sure that the past timestamps are actually correct
        lidar_pc = self.db.lidar_pc[token]
        timestamps = self.scenario.get_past_timestamps(iteration=iteration, time_horizon=time_horizon)
        timestamps.reverse()
        for i in range(len(timestamps)):
            lidar_pc = lidar_pc.prev
            self.assertEqual(lidar_pc.ego_pose.timestamp, timestamps[i].time_us)

    def test_timestamps(self) -> None:
        """
        Check that we can query the right timestamps
        """
        time_horizon = 1
        sampling_time = self.scenario.database_interval
        num_samples = int(time_horizon / sampling_time)
        iteration = 0
        start_time = self.scenario.start_time
        # Chang that num_samples is correctly computed
        timestamps_none = self.scenario.get_past_timestamps(iteration=iteration, time_horizon=time_horizon)

        # Make sure that time is increasing
        self.assertTrue(all(np.diff([ts.time_s for ts in timestamps_none]) > 0))
        self.assertAlmostEqual((start_time.time_s - timestamps_none[-1].time_s), sampling_time, delta=1e-2)

        # Check that all time steps are in the past
        self.assertTrue(all([timestamp.time_s - self.scenario.start_time.time_s < 0 for timestamp in timestamps_none]))
        timestamps_with = self.scenario.get_past_timestamps(
            iteration=iteration, time_horizon=time_horizon, num_samples=num_samples
        )
        self.assertEqual(len(timestamps_none), len(timestamps_with))

        # Make sure that also past samples from non zero iteration gives the right time
        timestamps_with = self.scenario.get_past_timestamps(
            iteration=100, time_horizon=time_horizon, num_samples=num_samples
        )
        self.assertTrue(all([timestamp.time_s - self.scenario.start_time.time_s > 0 for timestamp in timestamps_with]))

        # Extract all samples
        timestamps = self.scenario.get_future_timestamps(iteration=iteration, time_horizon=time_horizon)
        timestamps_s = [timestamp.time_s for timestamp in timestamps]
        self.assertTrue(
            all(
                [
                    np.isclose(time_diff, self.scenario.database_interval, atol=0.01)
                    for time_diff in np.diff(timestamps_s)
                ]
            )
        )

        # Half number of samples
        timestamps = self.scenario.get_future_timestamps(
            iteration=iteration, time_horizon=time_horizon, num_samples=int(time_horizon / 2 / sampling_time)
        )
        timestamps_s = [timestamp.time_s for timestamp in timestamps]
        self.assertTrue(
            all(
                [
                    np.isclose(time_diff, self.scenario.database_interval * 2, atol=0.02)
                    for time_diff in np.diff(timestamps_s)
                ]
            )
        )

    def test_all_functions_for_number_of_elements(self) -> None:
        """
        Test all functions which can query future and past from scenario
        """
        functions = [
            self.scenario.get_future_timestamps,
            self.scenario.get_past_timestamps,
            self.scenario.get_ego_past_trajectory,
            self.scenario.get_ego_future_trajectory,
            # self.scenario.get_past_tracked_objects, # This two are disabled because they are very slow
            # self.scenario.get_future_tracked_objects, # This two are disabled because they are very slow
        ]
        for function in functions:
            self.run_through_function(function)

    def run_through_function(
        self, fn: Union[Callable[[int, float, Optional[int]], List[Any]], Callable[[int, float], List[Any]]]
    ) -> None:
        """
        Test future and past queries with double the time step as the database has
        """
        # Future ego trajectory with double time step

        time_horizon = 10
        num_samples = 20
        sampling_time = self.scenario.database_interval

        # Test for all iteration within scenario
        for iteration in [0, 20, self.scenario.get_number_of_iterations() - 1]:
            # Check the number of elements in lists
            future_timestamps = fn(iteration, time_horizon)  # type: ignore
            self.assertEqual(len(future_timestamps), time_horizon / sampling_time)
            future_timestamps = fn(iteration, time_horizon - 0.01)  # type: ignore
            self.assertEqual(len(future_timestamps), time_horizon / sampling_time - 1)
            future_timestamps = fn(iteration, time_horizon + 0.01)  # type: ignore
            self.assertEqual(len(future_timestamps), time_horizon / sampling_time)
            future_timestamps = fn(iteration, time_horizon + sampling_time)  # type: ignore
            self.assertEqual(len(future_timestamps), time_horizon / sampling_time + 1)
            future_timestamps = fn(iteration, time_horizon, num_samples)  # type: ignore
            self.assertEqual(len(future_timestamps), num_samples)

    @unittest.skip('Flaky test')
    def test_get_route_roadblock_ids(self) -> None:
        """Test get_route_roadblock_ids"""
        roadblock_ids = self.scenario.get_route_roadblock_ids()

        if self.scenario.map_api and self.scenario.get_expert_ego_trajectory():
            self.assertGreater(len(roadblock_ids), 0)
            self.assertTrue(isinstance(roadblock_ids, List))
            self.assertTrue(isinstance(roadblock_ids[0], str))

        else:
            self.assertTrue(roadblock_ids is None)

    def test_get_traffic_light_statis_at_iteration(self) -> None:
        """Test get_traffic_light_status_at_iteration."""
        for iteration in range(self.scenario.get_number_of_iterations()):
            traffic_light_status = self.scenario.get_traffic_light_status_at_iteration(iteration=iteration)
            self.assertGreaterEqual(len(traffic_light_status), 0)


if __name__ == '__main__':
    unittest.main()
