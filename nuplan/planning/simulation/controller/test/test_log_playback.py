import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, PropertyMock, patch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

SCENARIO_EGO_STATE = "state"
CURRENT_ITERATION = 12
NEXT_ITERATION_IDX = 17


class TestLogPlaybackController(TestCase):
    """Tests implementation of LogPlaybackController"""

    def setUp(self) -> None:
        """
        Setup mocks for the tests
        """
        self.scenario = MagicMock(spec=AbstractScenario)
        self.iteration = Mock(spec=SimulationIteration)
        self.next_iteration = Mock(spec=SimulationIteration)
        self.ego_state = Mock(spec=EgoState)
        self.trajectory = Mock(spec=AbstractTrajectory)

        self.scenario.get_ego_state_at_iteration.return_value = SCENARIO_EGO_STATE
        self.next_iteration_idx = PropertyMock(return_value=NEXT_ITERATION_IDX)
        type(self.next_iteration).index = self.next_iteration_idx

        self.lpc = LogPlaybackController(self.scenario)

    @patch.object(LogPlaybackController, 'current_iteration', create=True, new_callable=PropertyMock)
    @patch.object(LogPlaybackController, 'scenario', create=True, new_callable=PropertyMock)
    def test_constructor(self, scenario: MagicMock, current_iteration: MagicMock) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        # Code execution
        LogPlaybackController(self.scenario)

        # Expectations check
        scenario.assert_called_once_with(self.scenario)
        current_iteration.assert_called_once_with(0)

    def test_get_state(self) -> None:
        """
        Tests if the scenario.get_ego_state_at_iteration is called with the current_iteration.
        """
        with patch.object(
            LogPlaybackController, 'current_iteration', create=True, new_callable=PropertyMock
        ) as current_iteration:
            current_iteration.return_value = CURRENT_ITERATION

            # Code execution
            result = self.lpc.get_state()

            # Expectations check
            self.assertEqual(result, SCENARIO_EGO_STATE)
            current_iteration.assert_called_once()
            self.scenario.get_ego_state_at_iteration.assert_called_once_with(CURRENT_ITERATION)

    def test_update_state(self) -> None:
        """
        Tests if the current_iteration is set to the next iteration.index value.
        """
        # Code execution
        with patch.object(
            LogPlaybackController, 'current_iteration', create=True, new_callable=PropertyMock
        ) as current_iteration:
            self.lpc.update_state(self.iteration, self.next_iteration, self.ego_state, self.trajectory)

            # Expectations check
            current_iteration.assert_called_once_with(NEXT_ITERATION_IDX)
            self.next_iteration_idx.assert_called_once()


if __name__ == '__main__':
    unittest.main()
