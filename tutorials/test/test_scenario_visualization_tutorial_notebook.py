import unittest
from os import path

from testbook import testbook

# to allow bazel to find tutorial file for testing
TUTORIAL_PATH_ABS = path.dirname(path.dirname(path.realpath(__file__)))


class TestScenarioVisualizationTutorialNotebook(unittest.TestCase):
    """
    Test scenario visualization tutorial Jupyter notebook across executed commands.
    """

    def test_scenario_visualization_execution(self) -> None:
        """
        Sanity test for notebook calling scenario visualization.
        """
        with testbook(path.join(TUTORIAL_PATH_ABS, 'nuplan_scenario_visualization.ipynb'), timeout=250) as tb:
            tb.execute_cell(range(7))


if __name__ == '__main__':
    unittest.main()
