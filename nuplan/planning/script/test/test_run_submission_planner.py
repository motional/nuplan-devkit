import unittest
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_submission_planner import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_simulation import SkeletonTestSimulation


class TestRunSubmissionPlanner(SkeletonTestSimulation):
    """Test running main submission planner."""

    @patch("nuplan.planning.script.run_submission_planner.SubmissionPlanner", autospec=True)
    def test_run_submission_planner(self, mock_submission_planner: Mock) -> None:
        """
        Sanity test to make sure hydra is setup correctly for run_submission_planner.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    'planner=simple_planner',
                ],
            )
            main(cfg)
            mock_submission_planner.assert_called_once()


if __name__ == '__main__':
    unittest.main()
