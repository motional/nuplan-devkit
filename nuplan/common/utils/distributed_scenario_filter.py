import logging
import os
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union, cast

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from pandas.errors import EmptyDataError

from nuplan.common.utils.file_backed_barrier import distributed_sync
from nuplan.common.utils.helpers import get_unique_job_id
from nuplan.common.utils.io_utils import safe_path_to_string
from nuplan.common.utils.s3_utils import check_s3_path_exists, expand_s3_dir, split_s3_path
from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import get_db_filenames_from_load_path
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list

logger = logging.getLogger(__name__)


class DistributedMode(Enum):
    """
    Constants to use to look up types of distributed processing for DistributedScenarioFilter
    They are:
    :param SCENARIO_BASED: Works in two stages, first getting a list of all, scenarios to process,
                           then breaking up that list and distributing across the workers
    :param LOG_FILE_BASED: Works in a single stage, breaking up the scenarios based on what log file they are in and
                           distributing the number of log files evenly across all workers
    :param SINGLE_NODE: Does no distribution, processes all scenarios in config
    """

    SCENARIO_BASED = "scenario_based"
    LOG_FILE_BASED = "log_file_based"
    SINGLE_NODE = "single_node"


class DistributedScenarioFilter:
    """
    Class to distribute the work to build / filter scenarios across workers, and to break up those scenarios in chunks to be
    handled on individual machines
    """

    def __init__(
        self,
        cfg: DictConfig,
        worker: WorkerPool,
        node_rank: int,
        num_nodes: int,
        synchronization_path: str,
        timeout_seconds: int = 7200,
        distributed_mode: DistributedMode = DistributedMode.SCENARIO_BASED,
    ):
        """
        :param cfg: top level config for the job (used to build scenario builder / scenario_filter)
        :param worker: worker to use in each node to parallelize the work
        :param node_rank: number from (0, num_nodes -1) denoting "which" node we are on
        :param num_nodes: total number of nodes the job is running on
        :param synchronization_path: path that can be in s3 or on a shared file system that will be used to synchronize
                                     across workers
        :param timeout_seconds: how long to wait during sync operations
        :param distributed_mode: what distributed mode to use to distribute computation
        """
        self._cfg = cfg
        self._worker = worker
        self._node_rank = node_rank
        self._num_nodes = num_nodes
        self.synchronization_path = synchronization_path
        self._timeout_seconds = timeout_seconds
        self._distributed_mode = distributed_mode

    def get_scenarios(self) -> List[AbstractScenario]:
        """
        Get all the scenarios that the current node should process
        :returns: list of scenarios for the current node
        """
        if self._num_nodes == 1 or self._distributed_mode == DistributedMode.SINGLE_NODE:
            logger.info("Building Scenarios in mode %s", DistributedMode.SINGLE_NODE)
            scenario_builder = build_scenario_builder(cfg=self._cfg)
            scenario_filter = build_scenario_filter(cfg=self._cfg.scenario_filter)
        elif self._distributed_mode in (DistributedMode.LOG_FILE_BASED, DistributedMode.SCENARIO_BASED):
            logger.info("Getting Log Chunks")
            current_chunk = self._get_log_db_files_for_single_node()

            logger.info("Getting Scenarios From Log Chunk of size %d", len(current_chunk))
            scenarios = self._get_scenarios_from_list_of_log_files(current_chunk)

            if self._distributed_mode == DistributedMode.LOG_FILE_BASED:
                logger.info(
                    "Distributed mode is %s, so we are just returning the scenarios"
                    "found from log files on the current worker.  There are %d scenarios to process"
                    "on node %d/%d",
                    DistributedMode.LOG_FILE_BASED,
                    len(scenarios),
                    self._node_rank,
                    self._num_nodes,
                )
                return scenarios

            logger.info(
                "Distributed mode is %s, so we are going to repartition the "
                "scenarios we got from the log files to better distribute the work",
                DistributedMode.SCENARIO_BASED,
            )
            logger.info("Getting repartitioned scenario tokens")
            tokens, log_db_files = self._get_repartition_tokens(scenarios)

            OmegaConf.set_struct(self._cfg, False)
            self._cfg.scenario_filter.scenario_tokens = tokens
            self._cfg.scenario_builder.db_files = log_db_files
            OmegaConf.set_struct(self._cfg, True)

            logger.info("Building repartitioned scenarios")
            scenario_builder = build_scenario_builder(cfg=self._cfg)
            scenario_filter = build_scenario_filter(cfg=self._cfg.scenario_filter)
        else:
            raise ValueError(
                f"Distributed mode must be one of "
                f"{[x.name for x in fields(DistributedMode)]}, "
                f"got {self._distributed_mode} instead!"
            )
        scenarios = scenario_builder.get_scenarios(scenario_filter, self._worker)
        return scenarios

    def _get_repartition_tokens(self, scenarios: List[AbstractScenario]) -> Tuple[List[str], List[str]]:
        """
        Submit list of scenarios found by the current node, sync up with other nodes to get the full list of tokens,
        and calculate the current node's set of tokens to process
        :param scenarios: Scenarios found by the current node
        :returns: (list of tokens, list of db files)
        """
        unique_job_id = get_unique_job_id()
        token_distribution_file_dir = Path(self.synchronization_path) / Path("tokens") / Path(unique_job_id)
        token_distribution_barrier_dir = Path(self.synchronization_path) / Path("barrier") / Path(unique_job_id)

        if self.synchronization_path.startswith("s3"):
            token_distribution_file_dir = safe_path_to_string(token_distribution_file_dir)
            token_distribution_barrier_dir = safe_path_to_string(token_distribution_barrier_dir)

        self._write_token_csv_file(scenarios, token_distribution_file_dir)
        distributed_sync(token_distribution_barrier_dir, timeout_seconds=self._timeout_seconds)

        token_distribution = self._get_all_generated_csv(token_distribution_file_dir)

        db_files_path = (
            Path(self._cfg.scenario_builder.db_files[0]).parent
            if isinstance(self._cfg.scenario_builder.db_files, (list, ListConfig))
            else Path(self._cfg.scenario_builder.db_files)
        )

        return self._get_token_and_log_chunk_on_single_node(token_distribution, db_files_path)

    def _get_all_generated_csv(self, token_distribution_file_dir: Union[Path, str]) -> List[Tuple[str, str]]:
        """
        Read the csv files that every machine in the cluster generated and get the full list of (token, db_file) pairs
        :param token_distribution_file_dir: path where to the csv files are stored
        :returns: full list of (token, db_file) pairs
        """
        if self.synchronization_path.startswith("s3"):
            token_distribution_file_list = [
                el for el in expand_s3_dir(token_distribution_file_dir) if el.endswith(".csv")
            ]
            token_distribution_list = []
            bucket, file_path = split_s3_path(Path(token_distribution_file_list[0]))
            s3_store = S3Store(s3_prefix=os.path.join('s3://', bucket))
            for token_distribution_file in token_distribution_file_list:
                with s3_store.get(token_distribution_file) as f:
                    try:
                        token_distribution_list.append(pd.read_csv(f, delimiter=','))
                    except EmptyDataError:
                        logger.warning(
                            "Token file for worker %s was empty, this may mean that something is wrong with your"
                            "configuration, or just that all of the data on that worker got filtered out.",
                            token_distribution_file,
                        )
        else:
            token_distribution_list = []
            for file_name in os.listdir(token_distribution_file_dir):
                try:
                    token_distribution_list.append(
                        pd.read_csv(os.path.join(token_distribution_file_dir, str(file_name)))
                    )
                except EmptyDataError:
                    logger.warning(
                        "Token file for worker %s was empty, this may mean that something is wrong with your"
                        "configuration, or just that all of the data on that worker got filtered out.",
                        file_name,
                    )

        if not token_distribution_list:
            raise AssertionError("No scenarios found to simulate!")

        token_distribution_df = pd.concat(token_distribution_list, ignore_index=True)

        token_distribution = token_distribution_df.values.tolist()

        return cast(List[Tuple[str, str]], token_distribution)

    def _get_token_and_log_chunk_on_single_node(
        self, token_distribution: List[Tuple[str, str]], db_files_path: Path
    ) -> Tuple[List[str], List[str]]:
        """
        Get the list of tokens and the list of logs those tokens are found in restricted to the current node
        :param token_distribution: Full list of all (token, log_file) pairs to be divided among the nodes
        :param db_files_path: Path to the actual db files
        """
        db_files_path_sanitized = safe_path_to_string(db_files_path)
        if not check_s3_path_exists(db_files_path_sanitized):
            raise AssertionError(
                f"Multinode caching only works in S3, but db_files path given was {db_files_path_sanitized}"
            )

        token_distribution_chunk = chunk_list(token_distribution, self._num_nodes)
        current_chunk = token_distribution_chunk[self._node_rank]
        current_logs_chunk = list({os.path.join(db_files_path_sanitized, f"{pair[1]}.db") for pair in current_chunk})
        current_token_chunk = [pair[0] for pair in current_chunk]

        return current_token_chunk, current_logs_chunk

    def _write_token_csv_file(
        self, scenarios: List[AbstractScenario], token_distribution_file_dir: Union[str, Path]
    ) -> None:
        """
        Writes a csv file of format token,log_name that stores the tokens associated with the given scenarios
        :param scenarios: Scenarios to take token/log pairs from
        :param token_distribution_file_dir: directory to write our csv file to
        """
        token_distribution_file = os.path.join(token_distribution_file_dir, f"{self._node_rank}.csv")
        token_log_pairs = [(scenario.token, scenario.log_name) for scenario in scenarios]
        os.makedirs(token_distribution_file_dir, exist_ok=True)
        token_log_pairs_df = pd.DataFrame(token_log_pairs)
        # The following to_csv function handles both local and s3 paths (via s3fs).
        token_log_pairs_df.to_csv(token_distribution_file, index=False)

    def _get_scenarios_from_list_of_log_files(self, log_db_files: List[str]) -> List[AbstractScenario]:
        """
        Gets the scenarios based on self._cfg, restricted to a list of log files
        :param log_db_files: list of log db files to restrict our search to
        :returns: list of scenarios
        """
        OmegaConf.set_struct(self._cfg, False)
        self._cfg.scenario_builder.db_files = log_db_files
        OmegaConf.set_struct(self._cfg, True)

        scenario_builder = build_scenario_builder(self._cfg)
        scenario_filter = build_scenario_filter(self._cfg.scenario_filter)
        scenarios: List[AbstractScenario] = scenario_builder.get_scenarios(scenario_filter, self._worker)
        return scenarios

    def _get_log_db_files_for_single_node(self) -> List[str]:
        """
        Get the list of log db files to be run on the current node
        :returns: list of log db files
        """
        if self._num_nodes == 1:
            return cast(List[str], self._cfg.scenario_builder.db_files)

        if not check_s3_path_exists(self._cfg.scenario_builder.db_files):
            raise AssertionError(
                f"DistributedScenarioFilter with multiple nodes "
                f"only works in S3, but db_files path given was {self._cfg.scenario_builder.db_files}"
            )

        all_files = get_db_filenames_from_load_path(self._cfg.scenario_builder.db_files)
        file_chunks = chunk_list(all_files, self._num_nodes)
        current_chunk = file_chunks[self._node_rank]

        return cast(List[str], current_chunk)
