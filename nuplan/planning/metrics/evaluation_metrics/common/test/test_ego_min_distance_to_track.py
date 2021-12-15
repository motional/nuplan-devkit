import unittest

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.metrics.evaluation_metrics.common.ego_min_distance_to_track import EgoMinDistanceToTrackStatistics
from nuplan.planning.metrics.metric_result import MetricStatisticsType, TimeSeries
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from pyquaternion import Quaternion


class TestEgoMinDistanceToTrack(unittest.TestCase):
    """ Run ego_min_distance_to_track unit tests. """

    def setUp(self) -> None:
        self.metric_name = "ego_min_distance_to_track"
        self.ego_min_distance_to_track = EgoMinDistanceToTrackStatistics(name=self.metric_name, category='Dynamics')
        self.history = self.setup_history()

    def setup_history(self) -> SimulationHistory:
        """ Set up a history. """

        # Mock Data
        scenario = MockAbstractScenario()
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())

        # Dummy 3d boxes.
        boxes_3d = [[Box3D(center=(1, 2, 0),
                           size=(1.0, 2.0, 1),
                           orientation=Quaternion(axis=(0, 0, 1), angle=-np.pi)),
                     Box3D(center=(3, 4, 0),
                           size=(1.0, 2.0, 1),
                           orientation=Quaternion(axis=(0, 0, 1), angle=-np.pi))
                     ],
                    [Box3D(center=(4, 5, 0),
                           size=(1.0, 2.0, 1),
                           orientation=Quaternion(axis=(0, 0, 1), angle=np.pi)),
                     Box3D(center=(6, 7, 0),
                           size=(1.0, 2.0, 1),
                           orientation=Quaternion(axis=(0, 0, 1), angle=np.pi))
                     ],
                    [Box3D(center=(8, 9, 0),
                           size=(1.0, 2.0, 1),
                           orientation=Quaternion(axis=(0, 0, 1), angle=np.pi / 2)),
                     Box3D(center=(10, 11, 0),
                           size=(1.0, 2.0, 1),
                           orientation=Quaternion(axis=(0, 0, 1), angle=np.pi / 2))
                     ]
                    ]

        ego_poses = [StateSE2(1, 2, np.pi),
                     StateSE2(2, 3, np.pi / 2),
                     StateSE2(3, 4, np.pi / 4)]

        values = [0.0, 1.5, 0.5]

        ego_states = [
            EgoState.from_raw_params(pose,
                                     velocity_2d=StateVector2D(value, 0),
                                     acceleration_2d=StateVector2D(value, 0),
                                     tire_steering_angle=0.0,
                                     time_point=TimePoint(t))
            for t, (pose, value) in enumerate(zip(ego_poses, values))]

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

        for ego_state, simulation_iteration, trajectory, boxes in zip(ego_states, simulation_iterations,
                                                                      trajectories, boxes_3d):
            history.add_sample(SimulationHistorySample(
                iteration=simulation_iteration,
                ego_state=ego_state,
                trajectory=trajectory,
                observation=Detections(boxes=boxes)
            ))

        return history

    def test_metric_name(self) -> None:
        """ Test metric name. """

        self.assertEqual(self.ego_min_distance_to_track.name, self.metric_name)

    def test_compute(self) -> None:
        """ Test compute. """

        results = self.ego_min_distance_to_track.compute(self.history)

        self.assertTrue(isinstance(results, list))
        metric = results[0]
        statistics = metric.statistics
        self.assertEqual(1.96, np.round(statistics[MetricStatisticsType.MAX].value, 2))  # type:ignore
        self.assertEqual(0.0, np.round(statistics[MetricStatisticsType.MIN].value, 2))  # type:ignore
        self.assertEqual(1.57, np.round(statistics[MetricStatisticsType.P90].value, 2))  # type:ignore

        time_series = metric.time_series
        assert isinstance(time_series, TimeSeries)
        self.assertEqual([0, 1, 2], time_series.time_stamps)
        self.assertEqual([0.0, 0.0, 1.96], np.round(time_series.values, 2).tolist())  # type:ignore


if __name__ == "__main__":
    unittest.main()
