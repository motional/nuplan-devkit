import logging
import os
import re
import shutil
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

from nuplan.common.utils.s3_utils import get_s3_client
from nuplan.planning.script.run_metric_aggregator import main as aggregator_main
from nuplan.planning.script.run_simulation import CONFIG_PATH
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.simulation.main_callback.publisher_callback import PublisherCallback
from nuplan.submission.evalai.leaderboard_writer import LeaderBoardWriter
from nuplan.submission.utils.aws_utils import s3_download
from nuplan.submission.utils.utils import get_submission_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
submission_logger = get_submission_logger(__name__)
# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

CONFIG_NAME = 'default_run_metric_aggregator'
NUM_INSTANCES_PER_CHALLENGE = 8
CHALLENGES = ['open_loop_boxes', 'closed_loop_nonreactive_agents', 'closed_loop_reactive_agents']


def is_submission_successful(challenges: List[str], simulation_results_dir: Path) -> bool:
    """
    Checks if evaluation of one submission was successful, by checking that all instances for all challenges
    were completed.
    :param challenges: The list of challenges.
    :param simulation_results_dir: Path were the simulation results are saved locally.
    :return: True if the submission was evaluated successfully, False otherwise.
    """
    completed = list(simulation_results_dir.rglob('*completed.txt'))
    successful = True if len(completed) == len(challenges) * NUM_INSTANCES_PER_CHALLENGE else False
    logger.info("Found %s completed simulations" % len(completed))
    logger.info("Simulation was successful:  %s" % successful)
    return successful


def list_subdirs_filtered(root_dir: Path, regex_pattern: re.Pattern[str]) -> List[str]:
    """
    Lists the path of files present in a directory. Results are filtered by ending pattern.
    :param root_dir: The path to start the search.
    :param regex_pattern: Regex based Pattern for which paths to keep.
    :return: List of paths under root_dir which wnd with path_end_filter.
    """
    paths = [
        str(path)
        for path in root_dir.rglob(
            '**/*',
        )
        if regex_pattern.search(str(path))
    ]
    return paths


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Downloads evaluation results from S3, runs metric aggregator and re-uploads the results.
    :param cfg: Hydra config dict.
    """
    # copy over the metric results from S3
    local_output_dir = Path(cfg.output_dir, cfg.contestant_id, cfg.submission_id)
    cfg.challenges = CHALLENGES

    Path(cfg.output_dir).mkdir(exist_ok=True, parents=True)
    s3_download(
        prefix='/'.join([cfg.contestant_id, cfg.submission_id]),
        local_path_name=cfg.output_dir,
        filters=None,
    )

    # Check if simulation was successful
    simulation_successful = is_submission_successful(cfg.challenges, local_output_dir)

    # Set up configuration
    cfg.output_dir = str(local_output_dir)
    cfg.scenario_metric_paths = list_subdirs_filtered(local_output_dir, re.compile(f'/{cfg.metric_folder_name}$'))
    logger.info("Found metric paths %s" % cfg.scenario_metric_paths)

    aggregated_metric_save_path = local_output_dir / cfg.aggregated_metric_folder_name

    leaderboard_writer = LeaderBoardWriter(cfg, str(local_output_dir))
    simulation_results = {}
    summary_results = {}
    try:
        if simulation_successful:
            shutil.rmtree(str(aggregated_metric_save_path), ignore_errors=True)
            aggregated_metric_save_path.mkdir(parents=True, exist_ok=True)

            aggregator_main(cfg)

            # Upload results
            simulation_results["aggregated-metrics"] = {
                "upload": True,
                "save_path": str(aggregated_metric_save_path),
                "remote_path": 'aggregated_metrics',
            }
            simulation_results["metrics"] = {
                "upload": True,
                "save_path": str(local_output_dir / cfg.metric_folder_name),
                "remote_path": 'metrics',
            }

            summary_results["summary"] = {
                "upload": True,
                "save_path": str(local_output_dir / 'summary'),
                "remote_path": "summary",
            }

    except Exception as e:
        submission_logger.error("Aggregation failed!")
        submission_logger.error(e)
        simulation_successful = False

    finally:
        simulation_results["submission_logs"] = {
            "upload": True,
            "save_path": "/tmp/submission.log",
            "remote_path": 'aggregated_metrics',
        }
        result_remote_prefix = [str(cfg.contestant_id), str(cfg.submission_id)]
        result_s3_client = get_s3_client()
        result_publisher_callback = PublisherCallback(
            simulation_results,
            remote_prefix=result_remote_prefix,
            s3_client=result_s3_client,
            s3_bucket=os.getenv("NUPLAN_SERVER_S3_ROOT_URL"),
        )
        result_publisher_callback.on_run_simulation_end()

        summary_publisher_callback = PublisherCallback(
            summary_results,
            remote_prefix=["public/leaderboard/planning/2022", cfg.submission_id],
            s3_client=result_s3_client,
            s3_bucket=os.getenv("NUPLAN_SERVER_S3_ROOT_URL"),
        )
        summary_publisher_callback.on_run_simulation_end()

    leaderboard_writer.write_to_leaderboard(simulation_successful=simulation_successful)

    # Cleanup
    shutil.rmtree(local_output_dir)


if __name__ == '__main__':
    main()
