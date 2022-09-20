import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from nuplan.submission.evalai.evalai_interface import EvalaiInterface

logger = logging.getLogger(__name__)


def read_metrics_from_results(results: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Transforms a pandas dataframe containing metric results to a string understandable by EvalAI leaderboard.
    :param results: The dataframes of metric results.
    :return: Dict holding the metric names and values.
    """
    ch1_df = results['open_loop_boxes']
    ch2_df = results['closed_loop_nonreactive_agents']
    ch3_df = results['closed_loop_reactive_agents']

    ch1, ch2, ch3 = [df.loc[df['scenario'] == 'final_score'] for df in [ch1_df, ch2_df, ch3_df]]

    metrics = {
        "ch1_overall_score": ch1['score'].values[0],
        "ch1_avg_displacement_error_within_bound": ch1['planner_expert_average_l2_error_within_bound'].values[0],
        "ch1_final_displacement_error_within_bound": ch1['planner_expert_final_l2_error_within_bound'].values[0],
        "ch1_miss_rate_within_bound": ch1['planner_miss_rate_within_bound'].values[0],
        "ch1_avg_heading_error_within_bound": ch1['planner_expert_average_heading_error_within_bound'].values[0],
        "ch1_final_heading_error_within_bound": ch1['planner_expert_final_heading_error_within_bound'].values[0],
        "ch2_overall_score": ch2['score'].values[0],
        "ch2_ego_is_making_progress": ch2['ego_is_making_progress'].values[0],
        "ch2_no_ego_at_fault_collisions": ch2['no_ego_at_fault_collisions'].values[0],
        "ch2_drivable_area_compliance": ch2['drivable_area_compliance'].values[0],
        "ch2_driving_direction_compliance": ch2['driving_direction_compliance'].values[0],
        "ch2_ego_is_comfortable": ch2['ego_is_comfortable'].values[0],
        "ch2_ego_progress_along_expert_route": ch2['ego_progress_along_expert_route'].values[0],
        "ch2_time_to_collision_within_bound": ch2['time_to_collision_within_bound'].values[0],
        "ch2_speed_limit_compliance": ch2['speed_limit_compliance'].values[0],
        "ch3_overall_score": ch3['score'].values[0],
        "ch3_ego_is_making_progress": ch3['ego_is_making_progress'].values[0],
        "ch3_no_ego_at_fault_collisions": ch3['no_ego_at_fault_collisions'].values[0],
        "ch3_drivable_area_compliance": ch3['drivable_area_compliance'].values[0],
        "ch3_driving_direction_compliance": ch3['driving_direction_compliance'].values[0],
        "ch3_ego_is_comfortable": ch3['ego_is_comfortable'].values[0],
        "ch3_ego_progress_along_expert_route": ch3['ego_progress_along_expert_route'].values[0],
        "ch3_time_to_collision_within_bound": ch3['time_to_collision_within_bound'].values[0],
        "ch3_speed_limit_compliance": ch3['speed_limit_compliance'].values[0],
        "combined_overall_score": np.mean([ch1['score'].values[0], ch2['score'].values[0], ch3['score'].values[0]]),
    }

    return metrics


class LeaderBoardWriter:
    """Class to write to EvalAI leaderboard."""

    def __init__(self, cfg: DictConfig, submission_path: str) -> None:
        """
        :param cfg: Hydra configuration
        :param submission_path: Path to the directory where the submission files are stored.
        """
        self.contestant_id = cfg.contestant_id
        self.submission_id = cfg.submission_id
        self.output_dir = cfg.output_dir
        self.aggregator_save_path = cfg.aggregator_save_path
        self.challenges = cfg.challenges

        with open(f'{submission_path}/submission_metadata.json', 'r') as file:
            self.submission_metadata = json.load(file)

        try:
            with open(f'{submission_path}/stout.log', 'r') as stdout:
                self.stdout = stdout.read()
        except FileNotFoundError:
            logger.info("No STDOUT log file found")
            self.stdout = ""

        try:
            with open(f'{submission_path}/stderr.log', 'r') as stderr:
                self.stderr = stderr.read()
        except FileNotFoundError:
            logger.info("No STDERR log file found")
            self.stderr = ""

        self.interface = EvalaiInterface()

    def write_to_leaderboard(self, simulation_successful: bool) -> None:
        """
        Writes to the leaderboard
        :param simulation_successful: Whether the simulation was successful or not.
        """
        if simulation_successful:
            logger.info("Writing to leaderboard SUCCESSFUL simulation...")
            data = self._on_successful_submission()
        else:
            logger.info("Writing to leaderboard FAILED simulation...")
            data = self._on_failed_submission()

        self.interface.update_submission_data(data)

    def _on_failed_submission(self) -> Dict[str, str]:
        """
        Builds leaderboard message for failed simulations.
        :return: Message to mark submission as failed
        """
        submission_data = {
            "challenge_phase": self.submission_metadata.get('challenge_phase'),
            "submission": self.submission_metadata.get('submission_id'),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "submission_status": "FAILED",
            "metadata": "",
        }
        return submission_data

    def _on_successful_submission(self) -> Dict[str, str]:
        """
        Builds leaderboard message for successful simulations.
        :return: Message to mark submission as successful, and to add metric values to leaderboard.
        """
        # Read metrics parquet
        results: Dict[str, pd.DataFrame] = {}  # Challenge to metrics parquet
        for challenge in self.challenges:
            challenge_result_files = Path(self.aggregator_save_path).glob('*.parquet')
            challenge_parquets = [pd.read_parquet(file) for file in challenge_result_files if challenge in str(file)]
            results[challenge] = challenge_parquets[0] if challenge_parquets else []

        # Construct json to update leaderboard
        result = json.dumps(
            [{"split": 'data_split', "show_to_participant": True, "accuracies": read_metrics_from_results(results)}]
        )

        submission_data = {
            "challenge_phase": self.submission_metadata.get('challenge_phase'),
            "submission": self.submission_metadata.get('submission_id'),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": result,
            "submission_status": "FINISHED",
            "metadata": {"status": "finished"},
        }
        return submission_data
