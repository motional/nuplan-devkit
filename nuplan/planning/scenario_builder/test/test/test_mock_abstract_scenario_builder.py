import unittest

from nuplan.common.utils.test_utils.interface_validation import assert_class_properly_implements_interface
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.test.mock_abstract_scenario_builder import MockAbstractScenarioBuilder


class TestMockAbstractScenarioBuilder(unittest.TestCase):
    """
    A class to test the MockAbstractScenarioBuilder utility class.
    """

    def test_mock_abstract_scenario_builder_implements_abstract_scenario_builder(self) -> None:
        """
        Tests that the mock abstract scenario builder class properly implements the interface.
        """
        assert_class_properly_implements_interface(AbstractScenarioBuilder, MockAbstractScenarioBuilder)


if __name__ == "__main__":
    unittest.main()
