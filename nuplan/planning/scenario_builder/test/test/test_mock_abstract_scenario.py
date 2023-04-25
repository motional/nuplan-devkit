import unittest

from nuplan.common.utils.test_utils.interface_validation import assert_class_properly_implements_interface
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario


class TestMockAbstractScenario(unittest.TestCase):
    """
    A class to test the MockAbstractScenario utility class.
    """

    def test_mock_abstract_scenario_implements_abstract_scenario(self) -> None:
        """
        Tests that the mock abstract scenario class properly implements the interface.
        """
        assert_class_properly_implements_interface(AbstractScenario, MockAbstractScenario)


if __name__ == "__main__":
    unittest.main()
