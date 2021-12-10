import unittest

import pytest
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.box import BoxObservation
from nuplan.planning.simulation.observation.lidar_pc import LidarPcObservation


class TestSimulationHistoryBuffer(unittest.TestCase):
    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = MockAbstractScenario(number_of_past_iterations=20)
        self.buffer_size = 10
        self._history_buffer = SimulationHistoryBuffer(buffer_size=self.buffer_size)

    def test_initialize_with_box(self) -> None:
        """ Test the initialize function """
        self.assertEqual(len(self._history_buffer), 0)
        box_observation = BoxObservation(self.scenario)
        self._history_buffer.initialize(scenario=self.scenario,
                                        observation_type=box_observation.observation_type())
        self.assertEqual(len(self._history_buffer), self.buffer_size)

    def test_initialize_with_lidar_pc(self) -> None:
        """ Test the initialize function """
        self.assertEqual(len(self._history_buffer), 0)
        lidar_pc_observation = LidarPcObservation(self.scenario)
        self._history_buffer.initialize(scenario=self.scenario,
                                        observation_type=lidar_pc_observation.observation_type())
        self.assertEqual(len(self._history_buffer), self.buffer_size)

    def test_append(self) -> None:
        """ Test the append function """
        self.assertEqual(len(self._history_buffer), 0)
        self._history_buffer.append(self.scenario.initial_ego_state, self.scenario.initial_detections)
        self.assertEqual(len(self._history_buffer), 1)

        self.assertEqual(self._history_buffer.ego_states, [self.scenario.initial_ego_state])
        self.assertEqual(self._history_buffer.observations, [self.scenario.initial_detections])

    def test_throw_on_empty_buffer(self) -> None:
        """ Test throw if buffer is empty """
        with pytest.raises(AttributeError):
            self._history_buffer.ego_states


if __name__ == '__main__':
    unittest.main()
