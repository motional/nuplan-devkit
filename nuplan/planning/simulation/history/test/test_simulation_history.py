import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample

DATA_LEN = 17


class TestSimulationHistory(TestCase):
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

    def test_clear(self) -> None:
        """
        Tests if the clear method clears the data list.
        """
        with patch.object(self.sh, 'data', clear=MagicMock()) as data:
            # Code execution
            self.sh.clear()

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


if __name__ == '__main__':
    unittest.main()
