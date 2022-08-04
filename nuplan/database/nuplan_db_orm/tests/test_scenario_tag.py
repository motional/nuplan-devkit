import unittest
from unittest.mock import Mock, PropertyMock, patch

from nuplan.database.nuplan_db_orm.scenario_tag import ScenarioTag


class TestScenarioTag(unittest.TestCase):
    """Tests class ScenarioTag"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.scenario_tag = ScenarioTag()

    @patch('nuplan.database.nuplan_db_orm.scenario_tag.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the session property"""
        # Setup
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock

        # Call the method under test
        result = self.scenario_tag._session

        # Assertions
        inspect_mock.assert_called_once_with(self.scenario_tag)
        self.assertEqual(result, session_mock)

    @patch('nuplan.database.nuplan_db_orm.scenario_tag.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the __repr__ method"""
        # Call the method under test
        result = self.scenario_tag.__repr__()

        # Assertions
        self.assertEqual(result, simple_repr_mock.return_value)


if __name__ == "__main__":
    unittest.main()
