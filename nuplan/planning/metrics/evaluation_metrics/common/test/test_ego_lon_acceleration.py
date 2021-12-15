import unittest

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration import EgoLonAccelerationStatistics
from nuplan.planning.metrics.metric_result import MetricStatisticsType, TimeSeries
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from pyquaternion import Quaternion


class TestEgoLonAcceleration(unittest.TestCase):
    """ Run ego_lon_acceleration unit tests. """

    def setUp(self) -> None:

        self.metric_name = "ego_lon_acceleration"
        self.ego_lon_acceleration_metric = EgoLonAccelerationStatistics(name=self.metric_name, category='Dynamics')
        self.history = self.setup_history()

    def setup_history(self) -> SimulationHistory:
        """ Set up a history. """

        # Mock Data
        scenario = MockAbstractScenario()
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())

        # Dummy 3d boxes.
        boxes = [Box3D(center=(664436.5810496865, 3997678.37696938, 0),
                       size=(1.8634377032974847, 4.555735325993202, 1),
                       orientation=Quaternion(axis=(0, 0, 1), angle=-1.50403628994573))]

        ego_states = [
            EgoState.from_raw_params(StateSE2(664430.1930625531, 3997650.6249544094, np.pi),
                                     velocity_2d=StateVector2D(x=0.0, y=0.0),
                                     acceleration_2d=StateVector2D(x=0.0, y=0.0),
                                     tire_steering_angle=0,
                                     time_point=TimePoint(0)),
            EgoState.from_raw_params(StateSE2(664431.1930625531, 3997651.6249544094, np.pi / 2),
                                     velocity_2d=StateVector2D(x=1.0, y=0.0),
                                     acceleration_2d=StateVector2D(x=1.0, y=0.0),
                                     tire_steering_angle=0,
                                     time_point=TimePoint(1)),
            EgoState.from_raw_params(StateSE2(664432.1930625531, 3997652.6249544094, np.pi / 4),
                                     velocity_2d=StateVector2D(x=0.5, y=0.0),
                                     acceleration_2d=StateVector2D(x=0.5, y=0.0),
                                     tire_steering_angle=0,
                                     time_point=TimePoint(2))
        ]

        simulation_iterations = [
            SimulationIteration(TimePoint(0), 0),
            SimulationIteration(TimePoint(1), 1),
            SimulationIteration(TimePoint(2), 2),
        ]

        trajectories = [
            InterpolatedTrajectory([ego_states[0], ego_states[1]]),
            InterpolatedTrajectory([ego_states[1], ego_states[2]]),
            InterpolatedTrajectory([ego_states[2], ego_states[2]]),
        ]

        for ego_state, simulation_iteration, trajectory in zip(ego_states, simulation_iterations, trajectories):
            history.add_sample(SimulationHistorySample(
                iteration=simulation_iteration,
                ego_state=ego_state,
                trajectory=trajectory,
                observation=Detections(boxes=boxes)
            ))

        return history

    def test_metric_name(self) -> None:
        """ Test metric name. """

        self.assertEqual(self.ego_lon_acceleration_metric.name, self.metric_name)

    def test_compute(self) -> None:
        """ Test compute. """

        results = self.ego_lon_acceleration_metric.compute(self.history)

        self.assertTrue(isinstance(results, list))
        metric = results[0]
        statistics = metric.statistics
        self.assertEqual(statistics[MetricStatisticsType.MAX].value, 1.0)
        self.assertEqual(statistics[MetricStatisticsType.MIN].value, 0)
        self.assertEqual(statistics[MetricStatisticsType.P90].value, 0.9)

        time_series = metric.time_series
        assert isinstance(time_series, TimeSeries)
        self.assertEqual(time_series.time_stamps, [0, 1, 2])
        self.assertEqual(time_series.values, [0.0, 1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
