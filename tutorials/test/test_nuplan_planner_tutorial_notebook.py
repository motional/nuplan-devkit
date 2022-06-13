import unittest
from os import environ, path

from testbook import testbook

# relative path to tutorial to pass configs to hydra in notebook
TUTORIAL_PATH_REL = path.dirname(path.dirname(path.relpath(__file__)))
# to allow bazel to find tutorial file for testing
TUTORIAL_PATH_ABS = path.dirname(path.dirname(path.realpath(__file__)))


class TestPlannerTutorialNotebook(unittest.TestCase):
    """
    Test planner tutorial Jupyter notebook across executed commands.
    """

    def test_full_execution(self) -> None:
        """
        Sanity test for notebook calling planner simulation.
        """
        environ['NUPLAN_TUTORIAL_PATH'] = TUTORIAL_PATH_REL
        with testbook(path.join(TUTORIAL_PATH_ABS, 'nuplan_planner_tutorial.ipynb'), timeout=250) as tb:
            # don't execute last (29th) cell due to non-breaking event loop is already running Runtime Error
            tb.execute_cell(range(28))


if __name__ == '__main__':
    unittest.main()
