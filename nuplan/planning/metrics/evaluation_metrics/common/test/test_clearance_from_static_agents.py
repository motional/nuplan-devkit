import os
import unittest

import hydra
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.metrics.metric_result import MetricStatisticsType
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from pyquaternion import Quaternion

CONFIG_PATH = os.path.join('..', '..', '..', '..', 'script/config/common/simulation_metric/common/')
CONFIG_NAME = 'clearance_from_static_agents_statistics'


class TestClearanceFromStaticAgents(unittest.TestCase):
    """ Run ego_acceleration unit tests. """

    def setUp(self) -> None:
        self.metric_name = "clearance_from_static_agents"

        # Hydra
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path=CONFIG_PATH)
        cfg = hydra.compose(config_name=CONFIG_NAME)

        self.clearance_metric = hydra.utils.instantiate(cfg)['clearance_from_static_agents_statistics']
        self.history = self.setup_history()

    def setup_history(self) -> SimulationHistory:
        """ Set up a history. """

        # Mock Data
        scenario = MockAbstractScenario()
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())

        # Dummy 3d boxes.
        boxes = [Box3D(center=(0.0, 5.0, 0.0),
                       size=(1.8634377032974847, 4.555735325993202, 1),
                       orientation=Quaternion(axis=(0, 0, 1), angle=0.0),
                       velocity=(0.0, 0.0, 0.0),
                       token="123456")]

        poses = [StateSE2(-10.0, -1.0, 0.0),
                 StateSE2(-5.0, 0.0, 0.0),
                 StateSE2(0.0, 0.0, 0.0),
                 StateSE2(5.0, -0.7, 0.0),
                 StateSE2(10.0, 1.3, 0.0)]

        ego_states = [EgoState.from_raw_params(pose,
                                               velocity_2d=StateVector2D(x=0.0, y=0.0),
                                               acceleration_2d=StateVector2D(x=0.0, y=0.0),
                                               tire_steering_angle=0,
                                               time_point=TimePoint(t)) for t, pose in enumerate(poses)]

        simulation_iterations = [SimulationIteration(TimePoint(t), t) for t in range(len(ego_states))]

        trajectories = [InterpolatedTrajectory([start_state, end_state]) for start_state, end_state in
                        zip(ego_states, ego_states[1:])]
        trajectories.append(InterpolatedTrajectory([ego_states[-1], ego_states[-1]]))

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

        self.assertEqual(self.clearance_metric.name, self.metric_name)

    def test_compute(self) -> None:
        """ Test compute. """

        results = self.clearance_metric.compute(self.history)

        self.assertTrue(isinstance(results, list))
        metric = results[0]
        statistics = metric.statistics
        self.assertAlmostEqual(3.92, statistics[MetricStatisticsType.MAX].value, 2)
        self.assertAlmostEqual(1.62, statistics[MetricStatisticsType.MIN].value, 2)
        self.assertAlmostEqual(3.80, statistics[MetricStatisticsType.P90].value, 2)


if __name__ == "__main__":
    unittest.main()
