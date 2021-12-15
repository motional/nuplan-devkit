import unittest
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.metrics.evaluation_metrics.common.ego_collisions import EgoCollisionStatistics
from nuplan.planning.metrics.metric_result import MetricStatisticsType
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from pyquaternion import Quaternion


@dataclass
class ExpectedCollisions:
    count: int
    collision_energies: List[float]


class TestEgoCollisions(unittest.TestCase):
    """ Run ego_acceleration unit tests. """

    def setUp(self) -> None:
        self.metric_name = "ego_collisions"
        self.ego_collision_count_metric = EgoCollisionStatistics(name=self.metric_name, category="Planning")

        self.ego_states, self.simulation_iterations, self.trajectories = self.mock_history_components()
        self.detections = self.mock_detections()

        # Map between sets of agents and expected number of collisions detected
        self.expected_collisions = {
            "agent_not_colliding": ExpectedCollisions(0, []),
            "agent_colliding_once": ExpectedCollisions(1, [0.5]),
            "agent_continuous_collision": ExpectedCollisions(1, [2.5]),
            "agent_colliding_twice": ExpectedCollisions(2, [5.0, 5.0]),
            "all": ExpectedCollisions(4, []),
        }

    def mock_detections(self) -> Dict[str, List[Box3D]]:
        """ Set up different test cases with fake detections. """
        number_time_steps = 4

        # Vehicle sizes
        box_length = 4.0
        box_width = 1.5
        box_height = 1.0

        # Vehicle orientation
        z_axis = (0, 0, 1)

        detections = {'agent_not_colliding': [Box3D(center=(10.0, 5.0, 0),
                                                    size=(box_width, box_length, box_height),
                                                    orientation=Quaternion(axis=z_axis,
                                                                           angle=-np.pi / 2),
                                                    velocity=(0.0, -2.0, 0.0),
                                                    token='0')] * number_time_steps,
                      'agent_colliding_once': [Box3D(center=(10.0, 0.0, 0),
                                                     size=(box_width, box_length, box_height),
                                                     orientation=Quaternion(axis=z_axis,
                                                                            angle=0.0),
                                                     velocity=(4.0, 0.0, 0.0),
                                                     token='1')] * number_time_steps,
                      'agent_continuous_collision': [Box3D(center=(8.0, 0.0, 0),
                                                           size=(box_width, box_length, box_height),
                                                           orientation=Quaternion(axis=z_axis,
                                                                                  angle=np.pi / 2),
                                                           velocity=(0.0, -0.0, 0.0),
                                                           token='2')] * number_time_steps}

        agent_colliding_twice = [Box3D(center=(0.0, 0.0, 0),
                                       size=(box_width, box_length, box_height),
                                       orientation=Quaternion(axis=z_axis, angle=np.pi),
                                       token='3',
                                       velocity=(0.0, -5.0, 0.0)) for _ in range(number_time_steps)]
        agent_colliding_twice[2].center = detections['agent_colliding_once'][0].center
        detections['agent_colliding_twice'] = agent_colliding_twice
        return detections

    def mock_history_components(self) -> Tuple[List[EgoState],
                                               List[SimulationIteration],
                                               List[InterpolatedTrajectory]]:
        """ Set up the components needed to build a history. """

        dynamic_car_state = DynamicCarState(get_pacifica_parameters().rear_axle_to_center,
                                            rear_axle_velocity_2d=StateVector2D(5.0, 0.0),
                                            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0))
        ego_poses = [StateSE2(0.0, 0.0, 0.0),
                     StateSE2(6.0, 0.0, 0.0),
                     StateSE2(10.0, 0.0, 0.0),
                     StateSE2(20.0, 0.0, 0.0)]

        ego_states = [EgoState(CarFootprint(pose), dynamic_car_state, 0.0, TimePoint(t)) for t, pose in
                      enumerate(ego_poses)]

        simulation_iterations = [
            SimulationIteration(TimePoint(0), 0),
            SimulationIteration(TimePoint(1), 1),
            SimulationIteration(TimePoint(2), 2),
            SimulationIteration(TimePoint(3), 3),
        ]

        trajectories = [
            InterpolatedTrajectory([ego_states[0], ego_states[1]]),
            InterpolatedTrajectory([ego_states[1], ego_states[2]]),
            InterpolatedTrajectory([ego_states[2], ego_states[3]]),
            InterpolatedTrajectory([ego_states[3], ego_states[3]]),
        ]

        return ego_states, simulation_iterations, trajectories

    def mock_history(self, all_detections: List[List[Box3D]]) -> SimulationHistory:
        """ Set up a mock history given a list of detections. """
        scenario = MockAbstractScenario()
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
        for ego_state, simulation_iteration, trajectory, detections in zip(self.ego_states, self.simulation_iterations,
                                                                           self.trajectories, all_detections):
            history.add_sample(SimulationHistorySample(
                iteration=simulation_iteration,
                ego_state=ego_state,
                trajectory=trajectory,
                observation=Detections(boxes=detections)
            ))
        return history

    def test_metric_name(self) -> None:
        """ Test metric name. """

        self.assertEqual(self.ego_collision_count_metric.name, self.metric_name)

    def test_collision_detections(self) -> None:
        """ Tests that the number of collisions is correct for various test cases. """

        for test_case, expected_collisions in self.expected_collisions.items():
            detections: List[List[Box3D]]
            if test_case == 'all':
                detections = [detections for detections in zip(*self.detections.values())]
            else:
                detections = [[detection] for detection in self.detections[test_case]]
            history = self.mock_history(detections)

            collision_violation_statistics = self.ego_collision_count_metric.compute(history)
            self.assertTrue(isinstance(collision_violation_statistics, list))

            statistics = collision_violation_statistics[0].statistics

            self.assertEqual(statistics[MetricStatisticsType.COUNT].value, expected_collisions.count,
                             f"Test case '{test_case}' has wrong number of collisions detected")
            if test_case != 'all' and statistics[MetricStatisticsType.COUNT].value > 0:
                self.assertAlmostEqual(statistics[MetricStatisticsType.MAX].value,
                                       max(expected_collisions.collision_energies), 3,
                                       f"Test case '{test_case}' has wrong energy of collisions detected")


if __name__ == "__main__":
    unittest.main()
