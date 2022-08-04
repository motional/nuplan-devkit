import unittest
from unittest.mock import Mock, PropertyMock, patch

from nuplan.database.nuplan_db_orm.traffic_light_status import TrafficLightStatus


class TestTrafficLightStatus(unittest.TestCase):
    """Tests the TrafficLightStatus class"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.traffic_light_status = TrafficLightStatus()

    @patch("nuplan.database.nuplan_db_orm.traffic_light_status.simple_repr", autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the repr method"""
        # Call the method under test
        result = self.traffic_light_status.__repr__()

        # Assertions
        simple_repr_mock.assert_called_once_with(self.traffic_light_status)
        self.assertEqual(result, simple_repr_mock.return_value)

    @patch("nuplan.database.nuplan_db_orm.traffic_light_status.inspect", autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the session property"""
        # Setup
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock

        # Call the method under test
        result = self.traffic_light_status._session()

        # Assertions
        inspect_mock.assert_called_once_with(self.traffic_light_status)
        self.assertEqual(result, session_mock.return_value)


if __name__ == "__main__":
    unittest.main()
