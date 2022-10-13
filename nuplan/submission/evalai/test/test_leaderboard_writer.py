import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from hydra import compose, initialize_config_dir

from nuplan.submission.evalai.leaderboard_writer import LeaderBoardWriter, read_metrics_from_results

CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../planning/script/config/simulation')
CONFIG_NAME = 'default_run_metric_aggregator'

TEST_FILE = 'nuplan.submission.evalai.leaderboard_writer'


class TestLeaderboardWriter(unittest.TestCase):
    """Tests for the LeaderboardWriter class."""

    @patch(f"{TEST_FILE}.EvalaiInterface")
    def setUp(self, mock_interface: Mock) -> None:
        """Sets up variables for testing."""
        self.mock_interface = mock_interface
        main_path = os.path.dirname(os.path.realpath(__file__))
        common_dir = 'file://' + os.path.join(main_path, '../../../planning/script/config/common')

        self.search_path = f'hydra.searchpath=[{common_dir}]'
        with initialize_config_dir(config_dir=CONFIG_PATH):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    self.search_path,
                    'contestant_id=contestant',
                    'submission_id=submission',
                ],
            )
            self.tmpdir = tempfile.TemporaryDirectory()
            self.addCleanup(self.tmpdir.cleanup)

            metadata = {
                'challenge_phase': 'phase',
                'submission_id': 'my_sub',
            }
            with open(f"{self.tmpdir.name}/submission_metadata.json", 'w') as fp:
                json.dump(metadata, fp)

            self.leaderboard_writer = LeaderBoardWriter(cfg, self.tmpdir.name)

    def test_write_to_leaderboard(self) -> None:
        """Tests that writing to leaderboard calls the correct callbacks an api."""
        with patch.object(self.leaderboard_writer, '_on_successful_submission'):
            self.leaderboard_writer.write_to_leaderboard(simulation_successful=True)
            self.leaderboard_writer._on_successful_submission.assert_called_once()
            self.leaderboard_writer.interface.update_submission_data.assert_called_once_with(
                self.leaderboard_writer._on_successful_submission.return_value
            )

        self.mock_interface.reset_mock()

        with patch.object(self.leaderboard_writer, '_on_failed_submission'):
            self.leaderboard_writer.write_to_leaderboard(simulation_successful=False)
            self.leaderboard_writer._on_failed_submission.assert_called_once()
            self.leaderboard_writer.interface.update_submission_data.assert_called_once_with(
                self.leaderboard_writer._on_failed_submission.return_value
            )

    def test__on_failed_submission(self) -> None:
        """Tests message creation on failes submission callback."""
        expected_data = {
            "challenge_phase": "phase",
            "submission": "my_sub",
            "stdout": "",
            "stderr": "",
            "submission_status": "FAILED",
            "metadata": "",
        }
        data = self.leaderboard_writer._on_failed_submission()
        self.assertEqual(expected_data, data)

    def test__on_successful_submission(self) -> None:
        """Tests message creation on successful submission callback."""
        expected_data = {
            "challenge_phase": "phase",
            "submission": "my_sub",
            "stdout": "",
            "stderr": "",
            "result": '[{"split": "data_split", "show_to_participant": true, ' '"accuracies": "results"}]',
            "submission_status": "FINISHED",
            "metadata": {'status': 'finished'},
        }
        with patch(f'{TEST_FILE}.read_metrics_from_results') as reader:
            reader.return_value = 'results'
            data = self.leaderboard_writer._on_successful_submission()
            self.assertEqual(expected_data, data)

    def test_read_metrics_from_results(self) -> None:
        """Tests parsing of dataframes."""
        dataframes = {
            'open_loop_boxes': pd.DataFrame.from_dict(
                {
                    'scenario': 'final_score',
                    'score': [0],
                    'planner_expert_average_l2_error_within_bound': [1],
                    'planner_expert_final_l2_error_within_bound': [2],
                    'planner_miss_rate_within_bound': [3],
                    'planner_expert_average_heading_error_within_bound': [4],
                    'planner_expert_final_heading_error_within_bound': [5],
                }
            ),
            'closed_loop_nonreactive_agents': pd.DataFrame.from_dict(
                {
                    'scenario': 'final_score',
                    'score': [10],
                    'ego_is_making_progress': [11],
                    'no_ego_at_fault_collisions': [12],
                    'drivable_area_compliance': [13],
                    'driving_direction_compliance': [14],
                    'ego_is_comfortable': [15],
                    'ego_progress_along_expert_route': [16],
                    'time_to_collision_within_bound': [17],
                    'speed_limit_compliance': [18],
                }
            ),
            'closed_loop_reactive_agents': pd.DataFrame.from_dict(
                {
                    'scenario': 'final_score',
                    'score': [110],
                    'ego_is_making_progress': [111],
                    'no_ego_at_fault_collisions': [112],
                    'drivable_area_compliance': [113],
                    'driving_direction_compliance': [114],
                    'ego_is_comfortable': [115],
                    'ego_progress_along_expert_route': [116],
                    'time_to_collision_within_bound': [117],
                    'speed_limit_compliance': [118],
                }
            ),
        }
        metrics = read_metrics_from_results(dataframes)
        expected_metrics = {
            'ch1_overall_score': 0,
            'ch1_avg_displacement_error_within_bound': 1,
            'ch1_final_displacement_error_within_bound': 2,
            'ch1_miss_rate_within_bound': 3,
            'ch1_avg_heading_error_within_bound': 4,
            'ch1_final_heading_error_within_bound': 5,
            'ch2_overall_score': 10,
            'ch2_ego_is_making_progress': 11,
            'ch2_no_ego_at_fault_collisions': 12,
            'ch2_drivable_area_compliance': 13,
            'ch2_driving_direction_compliance': 14,
            'ch2_ego_is_comfortable': 15,
            'ch2_ego_progress_along_expert_route': 16,
            'ch2_time_to_collision_within_bound': 17,
            'ch2_speed_limit_compliance': 18,
            'ch3_overall_score': 110,
            'ch3_ego_is_making_progress': 111,
            'ch3_no_ego_at_fault_collisions': 112,
            'ch3_drivable_area_compliance': 113,
            'ch3_driving_direction_compliance': 114,
            'ch3_ego_is_comfortable': 115,
            'ch3_ego_progress_along_expert_route': 116,
            'ch3_time_to_collision_within_bound': 117,
            'ch3_speed_limit_compliance': 118,
            'combined_overall_score': 40.0,
        }
        self.assertEqual(metrics, expected_metrics)


if __name__ == '__main__':
    unittest.main()
