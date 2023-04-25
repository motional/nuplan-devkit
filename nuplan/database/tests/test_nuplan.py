import unittest

from nuplan.database.tests.test_utils_nuplan_db import get_test_nuplan_db


class TestNuplan(unittest.TestCase):
    """Test Nuplan DB."""

    def test_nuplan(self) -> None:
        """
        Check whether the nuPlan DB can be loaded without errors.
        """
        db = get_test_nuplan_db()
        self.assertIsNotNone(db)


if __name__ == '__main__':
    unittest.main()
