import unittest
from unittest.mock import Mock, PropertyMock, patch

from nuplan.database.nuplan_db_orm.category import Category


class TestCategory(unittest.TestCase):
    """Test class Category"""

    def setUp(self) -> None:
        """
        Initializes a test Category
        """
        self.category = Category()

    @patch("nuplan.database.nuplan_db_orm.category.inspect", autospec=True)
    def test_session(self, inspect: Mock) -> None:
        """
        Tests _session methodtable property
        """
        # Setup
        mock_session = PropertyMock()
        inspect.return_value = Mock()
        inspect.return_value.session = mock_session

        # Call method under test
        result = self.category._session()

        # Assertions
        inspect.assert_called_once_with(self.category)
        mock_session.assert_called_once()
        self.assertEqual(result, mock_session.return_value)

    @patch("nuplan.database.nuplan_db_orm.category.simple_repr", autospec=True)
    def test_repr(self, simple_repr: Mock) -> None:
        """
        Tests string representation
        """
        # Call method under test
        result = self.category.__repr__()

        # Assertions
        simple_repr.assert_called_once_with(self.category)
        self.assertEqual(result, simple_repr.return_value)

    @patch("nuplan.database.nuplan_db_orm.category.default_color", autospec=True)
    def test_color(self, default_color: Mock) -> None:
        """
        Tests color property
        """
        # Call method under test
        result = self.category.color

        # Assertions
        default_color.assert_called_once_with(self.category.name)
        self.assertEqual(result, default_color.return_value)

    @patch("nuplan.database.nuplan_db_orm.category.default_color_np", autospec=True)
    def test_color_np(self, default_color_np: Mock) -> None:
        """
        Tests color_np property
        """
        # Call method under test
        result = self.category.color_np

        # Assertions
        default_color_np.assert_called_once_with(self.category.name)
        self.assertEqual(result, default_color_np.return_value)


if __name__ == "__main__":
    unittest.main()
