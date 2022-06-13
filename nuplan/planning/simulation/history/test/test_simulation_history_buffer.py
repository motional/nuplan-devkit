import unittest
from collections import deque
from unittest.mock import Mock

from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.lidar_pc import LidarPcObservation
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation


class TestSimulationHistoryBuffer(unittest.TestCase):
    """Test suite for SimulationHistoryBuffer"""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = MockAbstractScenario(number_of_past_iterations=20)
        self.buffer_size = 10

    def test_initialize_with_box(self) -> None:
        """Test the initialize function"""
        tracks_observation = TracksObservation(self.scenario)
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
            buffer_size=self.buffer_size, scenario=self.scenario, observation_type=tracks_observation.observation_type()
        )
        self.assertEqual(len(history_buffer), self.buffer_size)

    def test_initialize_with_lidar_pc(self) -> None:
        """Test the initialize function"""
        lidar_pc_observation = LidarPcObservation(self.scenario)
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
            buffer_size=self.buffer_size,
            scenario=self.scenario,
            observation_type=lidar_pc_observation.observation_type(),
        )
        self.assertEqual(len(history_buffer), self.buffer_size)

    def test_initialize_from_list(self) -> None:
        """Test the initialization from lists"""
        history_buffer = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=self.buffer_size,
            ego_states=[self.scenario.initial_ego_state],
            observations=[self.scenario.initial_tracked_objects],
            sample_interval=0.05,
        )
        self.assertEqual(len(history_buffer), 1)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state])
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects])

    def test_append(self) -> None:
        """Test the append function"""
        history_buffer = SimulationHistoryBuffer(
            ego_state_buffer=deque([Mock()], maxlen=1), observations_buffer=deque([Mock()], maxlen=1)
        )
        history_buffer.append(self.scenario.initial_ego_state, self.scenario.initial_tracked_objects)
        self.assertEqual(len(history_buffer), 1)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state])
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects])

    def test_extend(self) -> None:
        """Test the extend function"""
        history_buffer = SimulationHistoryBuffer(
            ego_state_buffer=deque([Mock()], maxlen=2), observations_buffer=deque([Mock()], maxlen=2)
        )
        history_buffer.extend([self.scenario.initial_ego_state] * 2, [self.scenario.initial_tracked_objects] * 2)

        self.assertEqual(len(history_buffer), 2)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state] * 2)
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects] * 2)


if __name__ == '__main__':
    unittest.main()
