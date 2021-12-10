import unittest
from os import environ, getenv, path

from testbook import testbook

# relative path to pass configs to hydra in notebook
TUTORIAL_PATH_REL = path.dirname(path.dirname(path.relpath(__file__)))
# to allow bazel to find file for testing
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
        with testbook(path.join(TUTORIAL_PATH_ABS, 'nuplan_planner_tutorial.ipynb')) as tb:
           tb.execute()

if __name__ == '__main__':
    unittest.main()
