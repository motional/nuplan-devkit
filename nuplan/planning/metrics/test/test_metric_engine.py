import unittest
from pathlib import Path

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.metrics.evaluation_metrics.common.ego_acceleration import EgoAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_jerk import EgoJerkStatistics
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.metrics.metric_result import TimeSeries
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class TestMetricEngine(unittest.TestCase):
    """Run metric_engine unit tests."""

    def setUp(self) -> None:
        """Set up a metric engine."""
        goal = StateSE2(x=664430.1930625531, y=3997650.6249544094, heading=0)
        self.scenario = MockAbstractScenario(mission_goal=goal)

        self.metric_names = ['ego_acceleration', 'ego_jerk']
        ego_acceleration_metric = EgoAccelerationStatistics(name=self.metric_names[0], category='Dynamics')

        ego_jerk = EgoJerkStatistics(name=self.metric_names[1], category='Dynamics', max_abs_mag_jerk=10.0)

        self.planner_name = 'planner'
        self.metric_engine = MetricsEngine(metrics=[ego_acceleration_metric], main_save_path=Path(''))
        self.metric_engine.add_metric(ego_jerk)
        self.history = self.setup_history()

    def setup_history(self) -> SimulationHistory:
        """Set up a history."""
        # Mock Data
        history = SimulationHistory(self.scenario.map_api, self.scenario.get_mission_goal())

        # Dummy scene_objects
        scene_objects = [
            SceneObject.from_raw_params(
                '1',
                '1',
                1,
                1,
                center=StateSE2(664436.5810496865, 3997678.37696938, -1.50403628994573),
                size=(1.8634377032974847, 4.555735325993202, 1.5),
            )
        ]
        vehicle_parameters = get_pacifica_parameters()
        ego_states = [
            EgoState.build_from_rear_axle(
                StateSE2(664430.3396621217, 3997673.373507501, -1.534863576938717),
                rear_axle_velocity_2d=StateVector2D(x=0.0, y=0.0),
                rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(1000000),
                vehicle_parameters=vehicle_parameters,
            ),
            EgoState.build_from_rear_axle(
                StateSE2(664431.1930625531, 3997675.37350750, -1.534863576938717),
                rear_axle_velocity_2d=StateVector2D(x=1.0, y=0.0),
                rear_axle_acceleration_2d=StateVector2D(x=0.5, y=0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(2000000),
                vehicle_parameters=vehicle_parameters,
            ),
            EgoState.build_from_rear_axle(
                StateSE2(664432.1930625531, 3997678.37350750, -1.534863576938717),
                rear_axle_velocity_2d=StateVector2D(x=0.5, y=0.0),
                rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(3000000),
                vehicle_parameters=vehicle_parameters,
            ),
            EgoState.build_from_rear_axle(
                StateSE2(664432.1930625531, 3997678.37350750, -1.534863576938717),
                rear_axle_velocity_2d=StateVector2D(x=0.5, y=0.0),
                rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(4000000),
                vehicle_parameters=vehicle_parameters,
            ),
            EgoState.build_from_rear_axle(
                StateSE2(664434.1930625531, 3997679.37350750, -1.534863576938717),
                rear_axle_velocity_2d=StateVector2D(x=0.5, y=0.0),
                rear_axle_acceleration_2d=StateVector2D(x=1.0, y=0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(5000000),
                vehicle_parameters=vehicle_parameters,
            ),
            EgoState.build_from_rear_axle(
                StateSE2(664434.1930625531, 3997679.37350750, -1.534863576938717),
                rear_axle_velocity_2d=StateVector2D(x=0.0, y=0.0),
                rear_axle_acceleration_2d=StateVector2D(x=2.0, y=0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(6000000),
                vehicle_parameters=vehicle_parameters,
            ),
        ]

        simulation_iterations = [
            SimulationIteration(TimePoint(1000000), 0),
            SimulationIteration(TimePoint(2000000), 1),
            SimulationIteration(TimePoint(3000000), 2),
            SimulationIteration(TimePoint(4000000), 3),
            SimulationIteration(TimePoint(5000000), 4),
        ]

        trajectories = [
            InterpolatedTrajectory([ego_states[0], ego_states[1]]),
            InterpolatedTrajectory([ego_states[1], ego_states[2]]),
            InterpolatedTrajectory([ego_states[2], ego_states[3]]),
            InterpolatedTrajectory([ego_states[3], ego_states[4]]),
            InterpolatedTrajectory([ego_states[4], ego_states[5]]),
        ]

        for ego_state, simulation_iteration, trajectory in zip(ego_states, simulation_iterations, trajectories):
            history.add_sample(
                SimulationHistorySample(
                    iteration=simulation_iteration,
                    ego_state=ego_state,
                    trajectory=trajectory,
                    observation=DetectionsTracks(TrackedObjects(scene_objects)),
                    traffic_light_status=self.scenario.get_traffic_light_status_at_iteration(
                        simulation_iteration.index
                    ),
                )
            )

        return history

    def test_compute(self) -> None:
        """Test compute() in MetricEngine."""
        expected_values = [[0.81, 0.04, 0.3, 0.81], [0.58, -0.28, 0.15, 0.58]]
        expected_time_stamps = [1000000, 2000000, 3000000, 4000000, 5000000]
        expected_time_series_values = [
            [0.21, 0.04, 0.09, 0.34, 0.81],
            [-0.28, -0.06, 0.15, 0.36, 0.58],
        ]
        metric_dict = self.metric_engine.compute(
            history=self.history, planner_name=self.planner_name, scenario=self.scenario
        )
        metric_files = metric_dict['mock_scenario_type_mock_scenario_name_planner']
        self.assertEqual(len(metric_files), 2)

        for index, metric_file in enumerate(metric_files):
            key = metric_file.key
            self.assertEqual(key.metric_name, self.metric_names[index])
            self.assertEqual(key.scenario_type, self.scenario.scenario_type)
            self.assertEqual(key.scenario_name, self.scenario.scenario_name)
            self.assertEqual(key.planner_name, self.planner_name)

            metric_statistics = metric_file.metric_statistics

            for statistic_result in metric_statistics:
                statistics = statistic_result.statistics
                self.assertEqual(np.round(statistics[0].value, 2), expected_values[index][0])
                self.assertEqual(np.round(statistics[1].value, 2), expected_values[index][1])
                self.assertEqual(np.round(statistics[2].value, 2), expected_values[index][2])
                self.assertEqual(np.round(statistics[3].value, 2), expected_values[index][3])

                time_series = statistic_result.time_series
                assert isinstance(time_series, TimeSeries)
                self.assertEqual(time_series.time_stamps, expected_time_stamps)
                self.assertEqual(np.round(time_series.values, 2).tolist(), expected_time_series_values[index])


if __name__ == '__main__':
    unittest.main()
