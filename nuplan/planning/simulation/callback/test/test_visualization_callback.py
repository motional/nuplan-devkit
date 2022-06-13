import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, PropertyMock, patch

from nuplan.planning.simulation.callback.visualization_callback import VisualizationCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.visualization.abstract_visualization import AbstractVisualization

TRAJECTORY = "test_trajectory"


class TestVisualizationCallback(TestCase):
    """Tests VisualizationCallback."""

    def setUp(self) -> None:
        """
        Setup mocks for the tests
        """
        self.visualization = Mock(spec=AbstractVisualization)
        self.setup = Mock(spec=SimulationSetup)
        self.planner = Mock(spec=AbstractPlanner)
        self.history = Mock(spec=SimulationHistory, data=[7, 23, 42])
        self.history_sample = Mock(spec=SimulationHistorySample)

        self.setup.scenario = "test_scenario"
        self.history_sample.ego_state = "test_ego_state"
        self.history_sample.observation = "test_observation"
        self.history_sample.iteration = "test_iteration"
        self.history_sample.trajectory = Mock()
        self.history_sample.trajectory.get_sampled_trajectory = Mock(return_value=TRAJECTORY)

        self.vc = VisualizationCallback(self.visualization)

        return super().setUp()

    @patch.object(VisualizationCallback, '_visualization', create=True, new_callable=PropertyMock)
    def test_constructor(self, visualization: MagicMock) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        # Code execution
        VisualizationCallback(self.visualization)

        # Expectations check
        visualization.assert_called_once_with(self.visualization)

    def test_on_initialization_start(self) -> None:
        """
        Tests if the visualization.render_scenario is called when the initialization starts.
        """
        with patch.object(self.vc, '_visualization', create=True, render_scenario=Mock()) as visualization:
            # Code execution
            self.vc.on_initialization_start(self.setup, self.planner)

            # Expectations check
            visualization.render_scenario.assert_called_once_with(self.setup.scenario, True)

    def test_on_step_end(self) -> None:
        """
        Tests if render_ego_state, render_observations, render_trajectory ,render
        are called with correct parameters in the on_step_end
        """
        with patch.object(self.vc, '_visualization', create=True) as visualization:
            visualization.render_ego_state = Mock()
            visualization.render_observations = Mock()
            visualization.render_trajectory = Mock()
            visualization.render = Mock()

            # Code execution
            self.vc.on_step_end(self.setup, self.planner, self.history_sample)

            # Expectations check
            visualization.render_ego_state.assert_called_once_with(self.history_sample.ego_state)
            visualization.render_observations.assert_called_once_with(self.history_sample.observation)
            visualization.render_trajectory.assert_called_once_with(TRAJECTORY)
            visualization.render.assert_called_once_with(self.history_sample.iteration)
            self.history_sample.trajectory.get_sampled_trajectory.assert_called_once()

    @patch.object(VisualizationCallback, 'on_step_end')
    def test_on_simulation_end(self, on_step_end: MagicMock) -> None:
        """
        Tests if on_step_end is called with correct parameters in the on_simulation_end
        """
        # Code execution
        self.vc.on_simulation_end(self.setup, self.planner, self.history)

        # Expectations check
        on_step_end.assert_called_once_with(self.setup, self.planner, self.history.data[-1])


if __name__ == '__main__':
    unittest.main()
