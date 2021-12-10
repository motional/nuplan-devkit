import os
import unittest

import hydra
from nuplan.actor_state.ego_state import EgoState
from nuplan.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.metrics.metric_result import MetricStatisticsType
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from pyquaternion import Quaternion

CONFIG_PATH = os.path.join('..', '..', '..', '..', 'script/config/common/simulation_metric/common/')
CONFIG_NAME = 'ego_is_comfortable_statistics'


class TestEgoIsComfortable(unittest.TestCase):
    """ Run ego is comfortable unit tests. """

    def setUp(self) -> None:

        self.metric_name = "ego_is_comfortable_statistics"

        # Hydra
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path=CONFIG_PATH)
        cfg = hydra.compose(config_name=CONFIG_NAME)

        self.ego_abs_jerk_magnitude = hydra.utils.instantiate(cfg)['ego_is_comfortable_statistics']
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

    def test_metric_name(self) -> None:
        """ Test metric name. """

        self.assertEqual(self.ego_abs_jerk_magnitude.name, self.metric_name)

    def test_compute(self) -> None:
        """ Test compute. """

        results = self.ego_abs_jerk_magnitude.compute(self.history)

        self.assertTrue(isinstance(results, list))
        metric = results[0]
        statistics = metric.statistics
        self.assertTrue(statistics[MetricStatisticsType.BOOLEAN])

        time_series = metric.time_series
        self.assertIsNone(time_series)


if __name__ == "__main__":
    unittest.main()
