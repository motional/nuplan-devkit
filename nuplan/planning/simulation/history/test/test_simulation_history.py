import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

DATA_LEN = 17


class TestSimulationHistory(TestCase):
    """Tests for SimulationHistory buffer."""

    def setUp(self) -> None:
        """
        Setup mocks for the tests
        """
        self.map = MagicMock(spec=AbstractMap)
        self.se2 = MagicMock(spec=StateSE2)
        self.sample = MagicMock(spec=SimulationHistorySample)
        self.sh = SimulationHistory(self.map, self.se2)

    def test_init(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        # Expectations check
        self.assertEqual(self.sh.map_api, self.map)
        self.assertEqual(self.sh.mission_goal, self.se2)
        self.assertEqual(self.sh.data, [])

    def test_add_sample(self) -> None:
        """
        Test if the add_sample method adds the passed sample to the data list.
        """
        with patch.object(self.sh, 'data', append=MagicMock()) as data:
            # Code execution
            self.sh.add_sample(self.sample)

            # Expectations check
            data.append.assert_called_once_with(self.sample)

    def test_last(self) -> None:
        """Test if the last method works as expected."""
        # Should raise an exception when data is None
        self.sh.data = None
        with self.assertRaises(RuntimeError):
            self.sh.last()

        # Nominal case
        self.sh.data = [self.sample, self.sample]
        self.assertEqual(self.sh.last(), self.sample)

    def test_extract_ego_state(self) -> None:
        """Test if the extract_ego_state property works as expected."""
        with patch('nuplan.planning.simulation.history.simulation_history.SimulationHistorySample', autospec=True):
            mock_data = [
                SimulationHistorySample(
                    iteration=MagicMock(),
                    ego_state=MagicMock(side_effect=lambda: i),
                    trajectory=MagicMock(),
                    observation=MagicMock(),
                    traffic_light_status=MagicMock(),
                )
                for i in range(DATA_LEN)
            ]
            self.sh.data = mock_data

            ego_states = self.sh.extract_ego_state
            self.assertTrue([ego_states[i]() == i for i in range(len(ego_states))])

    def test_clear(self) -> None:
        """
        Tests if the clear method clears the data list.
        """
        with patch.object(self.sh, 'data', clear=MagicMock()) as data:
            # Code execution
            self.sh.reset()

            # Expectations check
            data.clear.assert_called_once()

    @patch('nuplan.planning.simulation.history.simulation_history.len', return_value=DATA_LEN)
    def test_len(self, len_mock: MagicMock) -> None:
        """
        Tests if the len method returns the length of the data list.
        """
        # Code execution
        result = len(self.sh)

        # Expectations check
        self.assertEqual(result, DATA_LEN)
        len_mock.assert_called_once_with(self.sh.data)

    def test_interval_seconds(self) -> None:
        """Tests for the correct behavior of the interval_seconds property."""
        # Should raise an exception when data is None
        self.sh.data = None
        with self.assertRaises(ValueError):
            self.sh.interval_seconds

        # Should raise an exception when data is empty
        self.sh.data = []
        with self.assertRaises(ValueError):
            self.sh.interval_seconds

        with patch('nuplan.planning.simulation.history.simulation_history.SimulationHistorySample', autospec=True):
            # Should raise an exception when there's only a single data
            self.sh.data = [self.sample]
            with self.assertRaises(ValueError):
                self.sh.interval_seconds

            # Nominal case
            mock_data = [
                SimulationHistorySample(
                    iteration=SimulationIteration(index=0, time_point=TimePoint(i * 1e3)),
                    ego_state=MagicMock(),
                    trajectory=MagicMock(),
                    observation=MagicMock(),
                    traffic_light_status=MagicMock(),
                )
                for i in range(DATA_LEN)
            ]
            self.sh.data = mock_data

            expected_interval_seconds = mock_data[1].iteration.time_s - mock_data[0].iteration.time_s
            self.assertEqual(expected_interval_seconds, self.sh.interval_seconds)


if __name__ == '__main__':
    unittest.main()
