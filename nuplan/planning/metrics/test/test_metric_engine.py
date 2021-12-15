import unittest
from pathlib import Path

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.metrics.evaluation_metrics.common.ego_acceleration import EgoAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_distance_to_goal import EgoDistanceToGoalStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_jerk import EgoJerkStatistics
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.metrics.metric_result import MetricStatisticsType, TimeSeries
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from pyquaternion import Quaternion


class TestMetricEngine(unittest.TestCase):
    """ Run metric_engine unit tests. """

    def setUp(self) -> None:

        goal = StateSE2(x=664430.1930625531, y=3997650.6249544094, heading=0)
        self.scenario = MockAbstractScenario(mission_goal=goal)

        self.metric_names = ['ego_acceleration', 'ego_distance_to_goal', 'ego_jerk']
        ego_acceleration_metric = EgoAccelerationStatistics(name=self.metric_names[0], category='Dynamics')

        ego_distance_to_goal = EgoDistanceToGoalStatistics(name=self.metric_names[1], category='Planning')

        ego_jerk = EgoJerkStatistics(name=self.metric_names[2], category='Dynamics')

        self.scenario_name = 'Scenario'
        self.scenario_type = 'Scenario_type'
        self.planner_name = 'planner'
        self.metric_engine = MetricsEngine(scenario_type=self.scenario_type,
                                           metrics=[ego_acceleration_metric, ego_distance_to_goal],
                                           main_save_path=Path(''), timestamp=0)
        self.metric_engine.add_metric(ego_jerk)
        self.history = self.setup_history()

    def setup_history(self) -> SimulationHistory:
        """ Set up a history. """

        # Mock Data
        history = SimulationHistory(self.scenario.map_api, self.scenario.get_mission_goal())

        # Dummy 3d boxes.
        boxes = [Box3D(center=(664436.5810496865, 3997678.37696938, 0),
                       size=(1.8634377032974847, 4.555735325993202, 1),
                       orientation=Quaternion(axis=(0, 0, 1), angle=-1.50403628994573))]

        ego_states = [
            EgoState.from_raw_params(StateSE2(664430.3396621217, 3997673.373507501, -1.534863576938717),
                                     velocity_2d=StateVector2D(x=0.0, y=0.0),
                                     acceleration_2d=StateVector2D(x=0.0, y=0.0),
                                     tire_steering_angle=0.0,
                                     time_point=TimePoint(1000000)),
            EgoState.from_raw_params(StateSE2(664431.1930625531, 3997675.37350750, -1.534863576938717),
                                     velocity_2d=StateVector2D(x=1.0, y=0.0),
                                     acceleration_2d=StateVector2D(x=0.5, y=0.0),
                                     tire_steering_angle=0.0,
                                     time_point=TimePoint(2000000)),
            EgoState.from_raw_params(StateSE2(664432.1930625531, 3997678.37350750, -1.534863576938717),
                                     velocity_2d=StateVector2D(x=0.5, y=0.0),
                                     acceleration_2d=StateVector2D(x=0.0, y=0.0),
                                     tire_steering_angle=0.0,
                                     time_point=TimePoint(3000000)),
            EgoState.from_raw_params(StateSE2(664434.1930625531, 3997679.37350750, -1.534863576938717),
                                     velocity_2d=StateVector2D(x=0.5, y=0.0),
                                     acceleration_2d=StateVector2D(x=1.0, y=0.0),
                                     tire_steering_angle=0.0,
                                     time_point=TimePoint(4000000)),
            EgoState.from_raw_params(StateSE2(664434.1930625531, 3997679.37350750, -1.534863576938717),
                                     velocity_2d=StateVector2D(x=0.0, y=0.0),
                                     acceleration_2d=StateVector2D(x=2.0, y=0.0),
                                     tire_steering_angle=0.0,
                                     time_point=TimePoint(5000000)),
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
            InterpolatedTrajectory([ego_states[2], ego_states[2]]),
            InterpolatedTrajectory([ego_states[3], ego_states[4]]),
            InterpolatedTrajectory([ego_states[3], ego_states[4]]),
        ]

        for ego_state, simulation_iteration, trajectory in zip(ego_states, simulation_iterations, trajectories):
            history.add_sample(SimulationHistorySample(
                iteration=simulation_iteration,
                ego_state=ego_state,
                trajectory=trajectory,
                observation=Detections(boxes=boxes)
            ))

        return history

    def test_compute(self) -> None:
        """ Test compute() in MetricEngine. """

        expected_values = [
            [2.0, 0.0, 1.6],
            [29.03, 22.75, 29.03],
            [1.16, -0.26, 1.02]
        ]
        expected_time_stamps = [1000000, 2000000, 3000000, 4000000, 5000000]
        expected_time_series_values = [
            [0.0, 0.5, 0.0, 1.0, 2.0],
            [22.75, 24.77, 27.82, 29.03, 29.03],
            [-0.26, 0.09, 0.45, 0.81, 1.16]
        ]
        metric_files = self.metric_engine.compute(
            history=self.history, planner_name=self.planner_name, scenario_name=self.scenario_name)

        self.assertEqual(len(metric_files), 3)

        for index, metric_file in enumerate(metric_files):
            key = metric_file.key
            self.assertEqual(key.metric_name, self.metric_names[index])
            self.assertEqual(key.scenario_type, self.scenario_type)
            self.assertEqual(key.scenario_name, self.scenario_name)
            self.assertEqual(key.planner_name, self.planner_name)

            metric_statistics = metric_file.metric_statistics

            for statistics_group_name, statistic_results in metric_statistics.items():
                for statistic_result in statistic_results:
                    statistics = statistic_result.statistics
                    self.assertEqual(np.round(statistics[MetricStatisticsType.MAX].value, 2),  # type:ignore
                                     expected_values[index][0])
                    self.assertEqual(np.round(statistics[MetricStatisticsType.MIN].value, 2),  # type:ignore
                                     expected_values[index][1])
                    self.assertEqual(np.round(statistics[MetricStatisticsType.P90].value, 2),  # type:ignore
                                     expected_values[index][2])

                    time_series = statistic_result.time_series
                    assert isinstance(time_series, TimeSeries)
                    self.assertEqual(time_series.time_stamps, expected_time_stamps)
                    self.assertEqual(np.round(time_series.values, 2).tolist(),  # type:ignore
                                     expected_time_series_values[index])
