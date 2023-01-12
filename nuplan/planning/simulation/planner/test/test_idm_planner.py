import unittest
from unittest.mock import MagicMock, Mock, call, patch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class TestAbstractIDMPlanner(unittest.TestCase):
    """Test the AbstractIDMPlanner interface"""

    TEST_FILE_PATH = "nuplan.planning.simulation.planner.idm_planner"

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self.scenario = get_test_nuplan_scenario()
        self.planned_trajectory_samples = 10
        self.planner = IDMPlanner(
            target_velocity=10,
            min_gap_to_lead_agent=0.5,
            headway_time=1.5,
            accel_max=1.0,
            decel_max=2.0,
            planned_trajectory_samples=self.planned_trajectory_samples,
            planned_trajectory_sample_interval=0.2,
            occupancy_map_radius=20,
        )

    def test_name(self) -> None:
        """Test name"""
        self.assertEqual(self.planner.name(), "IDMPlanner")

    def test_observation_type(self) -> None:
        """Test observation_type"""
        self.assertEqual(self.planner.observation_type(), DetectionsTracks)

    def test__initialize_route_plan_assertion_error(self) -> None:
        """Test raise if _map_api is uninitialized"""
        with self.assertRaises(AssertionError):
            self.planner._initialize_route_plan([])

    @patch(f"{TEST_FILE_PATH}.IDMPlanner._initialize_route_plan")
    def test_initialize(self, mock_initialize_route_plan: Mock) -> None:
        """Test initialize"""
        initialization = MagicMock()
        self.planner.initialize(initialization)
        mock_initialize_route_plan.assert_called_once_with(initialization.route_roadblock_ids)

    @patch(f"{TEST_FILE_PATH}.path_to_linestring")
    @patch(f"{TEST_FILE_PATH}.create_path_from_se2")
    @patch(f"{TEST_FILE_PATH}.IDMPlanner._breadth_first_search")
    @patch(f"{TEST_FILE_PATH}.IDMPlanner._get_starting_edge")
    def test__initialize_ego_path(
        self,
        mock_get_starting_edge: Mock,
        mock_breadth_first_search: Mock,
        mock_create_path_from_se2: Mock,
        mock_path_to_linestring: Mock,
    ) -> None:
        """Test _initialize_ego_path()"""
        mock_starting_edge = Mock()
        mock_lane = MagicMock()
        mock_lane.speed_limit_mps = 0
        ego_state = self.scenario.initial_ego_state
        mock_breadth_first_search.return_value = ([mock_lane], True)
        mock_get_starting_edge.return_value = mock_starting_edge
        with patch.object(self.planner, "_route_roadblocks"):
            self.planner._initialize_ego_path(ego_state)

            mock_breadth_first_search.assert_called_once_with(ego_state)
            mock_create_path_from_se2.assert_called_once_with([])
            mock_path_to_linestring.assert_called_once_with([])

    def test__get_starting_edge(self) -> None:
        """Test _get_starting_edge()"""
        mock_edge = MagicMock(spec_set=LaneGraphEdgeMapObject)
        mock_edge.contains_point.side_effect = [False, True]
        mock_edge.polygon.distance.side_effect = [0, 0]
        mock_roadblock = MagicMock(spec_set=RoadBlockGraphEdgeMapObject)
        mock_roadblock.interior_edges = [mock_edge]
        self.planner._route_roadblocks = [mock_roadblock, mock_roadblock]

        result = self.planner._get_starting_edge(Mock(spec=EgoState))
        mock_edge.contains_point.assert_called()
        mock_edge.polygon.distance.assert_called()
        self.assertEqual(result, mock_edge)

    @patch(f"{TEST_FILE_PATH}.IDMPlanner._initialize_ego_path")
    @patch(f"{TEST_FILE_PATH}.IDMPlanner._construct_occupancy_map")
    @patch(f"{TEST_FILE_PATH}.IDMPlanner._annotate_occupancy_map")
    @patch(f"{TEST_FILE_PATH}.IDMPlanner._get_planned_trajectory")
    def test_compute_trajectory(
        self,
        mock_get_planned_trajectory: Mock,
        mock_annotate_occupancy_map: Mock,
        mock_construct_occupancy_map: Mock,
        mock_initialize_ego_path: Mock,
    ) -> None:
        """Test compute_trajectory"""
        # Create mocks for List[PlannerInput]
        planner_input = MagicMock()
        mock_ego_state = Mock()
        mock_traffic_light_data = call()
        planner_input.history.current_state = (mock_ego_state, Mock())
        planner_input.traffic_light_data = mock_traffic_light_data

        # Create mocks for _construct_occupancy_map
        mock_occupancy_map = Mock()
        mock_unique_observations = Mock()
        mock_construct_occupancy_map.return_value = (mock_occupancy_map, mock_unique_observations)

        self.planner.compute_trajectory(planner_input)
        mock_initialize_ego_path.assert_called_once_with(mock_ego_state)
        mock_construct_occupancy_map.assert_called_once_with(*planner_input.history.current_state)
        mock_annotate_occupancy_map.assert_called_once_with(mock_traffic_light_data, mock_occupancy_map)
        mock_get_planned_trajectory.assert_called_once_with(
            mock_ego_state, mock_occupancy_map, mock_unique_observations
        )

    def test_compute_trajectory_integration(self) -> None:
        """Test the IDMPlanner in full using mock data"""
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(10, self.scenario, DetectionsTracks)

        self.planner.initialize(
            PlannerInitialization(
                self.scenario.get_route_roadblock_ids(),
                self.scenario.get_mission_goal(),
                self.scenario.map_api,
            )
        )
        trajectories = self.planner.compute_trajectory(
            PlannerInput(
                SimulationIteration(self.scenario.get_time_point(0), 0),
                history_buffer,
                list(self.scenario.get_traffic_light_status_at_iteration(0)),
            )
        )

        # Plus 1 because the planner should append it's current state
        self.assertEqual(self.planned_trajectory_samples + 1, len(trajectories.get_sampled_trajectory()))


if __name__ == '__main__':
    unittest.main()
