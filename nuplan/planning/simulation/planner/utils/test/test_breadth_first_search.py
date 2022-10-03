import unittest
from unittest.mock import MagicMock, Mock, patch

from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch


class TestBreadthFirstSearch(unittest.TestCase):
    """Test class for BreadthFirstSearch"""

    TEST_FILE_PATH = "nuplan.planning.simulation.planner.utils.breadth_first_search"

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self._mock_edge = MagicMock(spec=LaneGraphEdgeMapObject)
        self._graph_search = BreadthFirstSearch(self._mock_edge, ["a"])
        self._mock_edge.id = "a"
        self._mock_edge.get_roadblock_id.side_effect = ["1", "2", "3"]
        self._mock_edge.outgoing_edges = [self._mock_edge]

    @patch(f"{TEST_FILE_PATH}.BreadthFirstSearch._check_end_condition")
    @patch(f"{TEST_FILE_PATH}.BreadthFirstSearch._check_goal_condition")
    @patch(f"{TEST_FILE_PATH}.BreadthFirstSearch._construct_path")
    def test__breadth_first_search(
        self, mock_construct_path: Mock, mock_check_goal_condition: Mock, mock_check_end_condition: Mock
    ) -> None:
        """Test search()"""
        mock_check_goal_condition.side_effect = [False, False, True]
        mock_check_end_condition.side_effect = [False, False, False, False, False]
        _, path_found = self._graph_search.search(Mock(), 3)
        self.assertTrue(path_found)
        mock_check_goal_condition.assert_called()
        mock_check_end_condition.assert_called()
        mock_construct_path.assert_called_once()

    def test__construct_path(self) -> None:
        """Test _construct_path()"""
        mock_edge_1 = MagicMock(spec=LaneGraphEdgeMapObject)
        mock_edge_1.id = "a"
        mock_edge_2 = MagicMock(spec=LaneGraphEdgeMapObject)
        mock_edge_2.id = "b"
        self._graph_search._parent = {"a_2": mock_edge_2, "b_1": None}
        path = self._graph_search._construct_path(mock_edge_1, 2)
        self.assertEqual(path, [mock_edge_2, mock_edge_1])

    def test__check_end_condition(self) -> None:
        """Test _check_end_condition()"""
        self.assertTrue(self._graph_search._check_end_condition(1, 0))
        self.assertFalse(self._graph_search._check_end_condition(1, 1))

    def test__check_goal_condition(self) -> None:
        """Test _check_goal_condition()"""
        mock_edge = MagicMock(spec=LaneGraphEdgeMapObject)
        mock_edge.get_roadblock_id.side_effect = ["1", "2", "3", "3"]
        mock_target_block = MagicMock(spec=LaneGraphEdgeMapObject)
        mock_target_block.id = "3"

        self.assertFalse(self._graph_search._check_goal_condition(mock_edge, mock_target_block, 0, 3))
        self.assertFalse(self._graph_search._check_goal_condition(mock_edge, mock_target_block, 3, 3))
        self.assertFalse(self._graph_search._check_goal_condition(mock_edge, mock_target_block, 0, 3))
        self.assertTrue(self._graph_search._check_goal_condition(mock_edge, mock_target_block, 3, 3))


if __name__ == '__main__':
    unittest.main()
