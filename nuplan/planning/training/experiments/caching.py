import gc
import itertools
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from nuplan.common.utils.file_backed_barrier import FileBackedBarrier
from nuplan.common.utils.helpers import get_unique_job_id
from nuplan.common.utils.s3_utils import check_s3_path_exists, expand_s3_dir, split_s3_path
from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import (
    discover_log_dbs,
    get_db_filenames_from_load_path,
)
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    sanitize_s3_path,
    save_cache_metadata,
)
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list, worker_map

logger = logging.getLogger(__name__)


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """
    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_internal(args: List[Dict[str, Union[List[AbstractScenario], DictConfig]]]) -> List[CacheResult]:
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        scenarios: List[AbstractScenario] = [a["scenario"] for a in args]
        cfg: DictConfig = args[0]["cfg"]

        model = build_torch_module_wrapper(cfg.model)
        feature_builders = model.get_list_of_required_feature()
        target_builders = model.get_list_of_computed_target()

        # Now that we have the feature and target builders, we do not need the model any more.
        # Delete it so it gets gc'd and we can save a few system resources.
        del model

        # Create feature preprocessor
        assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"
        preprocessor = FeaturePreprocessor(
            cache_path=cfg.cache.cache_path,
            force_feature_computation=cfg.cache.force_feature_computation,
            feature_builders=feature_builders,
            target_builders=target_builders,
        )

        logger.info("Extracted %s scenarios for thread_id=%s, node_id=%s.", str(len(scenarios)), thread_id, node_id)
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        for idx, scenario in enumerate(scenarios):
            logger.info(
                "Processing scenario %s / %s in thread_id=%s, node_id=%s",
                idx + 1,
                len(scenarios),
                thread_id,
                node_id,
            )

            features, targets, file_cache_metadata = preprocessor.compute_features(scenario)

            scenario_num_failures = sum(
                0 if feature.is_valid else 1 for feature in itertools.chain(features.values(), targets.values())
            )
            scenario_num_successes = len(features.values()) + len(targets.values()) - scenario_num_failures
            num_failures += scenario_num_failures
            num_successes += scenario_num_successes
            all_file_cache_metadata += file_cache_metadata

        logger.info("Finished processing scenarios for thread_id=%s, node_id=%s", thread_id, node_id)
        return [CacheResult(failures=num_failures, successes=num_successes, cache_metadata=all_file_cache_metadata)]

    result = cache_scenarios_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result


def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

    # Split the files and edit cfg to only contain the file chunks to be handled by current node in multinode setting
    current_chunk = split_and_extract_file_chunks_for_current_node(cfg)

    # None is possible when running locally, and means to use all files in data_root
    if current_chunk is None:
        current_chunk = cfg.scenario_builder.data_root

    log_db_files = discover_log_dbs(current_chunk)

    if int(os.environ.get("NUM_NODES", 1)) > 1 and cfg.distribute_by_scenario:
        scenarios = get_all_scenarios_on_machine(log_db_files, worker, cfg)
        current_token_chunk, current_logs_chunk = get_repartition_tokens(scenarios, cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.scenario_builder.db_files = current_logs_chunk
        cfg.scenario_filter.scenario_tokens = current_token_chunk
        cfg.scenario_filter.limit_total_scenarios = None  # the filtering already happened before
        OmegaConf.set_struct(cfg, True)

    logger.info(
        "Found %s db files, containing %s scenarios to process. Extracting scenarios and building features and targets.",
        str(len(cfg.scenario_builder.db_files)) if cfg.scenario_builder.db_files is not None else "None",
        str(len(cfg.scenario_filter.scenario_tokens)) if cfg.scenario_filter.scenario_tokens is not None else "None",
    )

    scenario_builder = build_scenario_builder(cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

    data_points = [{"scenario": scenario, "cfg": cfg} for scenario in scenarios]
    logger.info("Starting dataset caching of %s files...", str(len(data_points)))

    cache_results = worker_map(worker, cache_scenarios, data_points)

    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    logger.info("Completed dataset caching! Failed features and targets: %s out of %s", str(num_fail), str(num_total))

    cached_metadata = [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]

    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info(f"Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets...")
    save_cache_metadata(cached_metadata, Path(cfg.cache.cache_path), node_id)
    logger.info("Done storing metadata csv file.")


def split_and_extract_file_chunks_for_current_node(cfg: DictConfig) -> List[str]:
    """
    Splits the list of .db files into equal chunks and edits the cfg
    to only contain the chunk of files relevant to the current node
    :param cfg: Omegaconf dictionary
    :return: List of .db files relevant to current node
    """
    num_nodes = int(os.environ.get("NUM_NODES", 1))
    logger.info("Number of Nodes used: %s", str(num_nodes))
    if num_nodes == 1:
        return cast(List[str], cfg.scenario_builder.db_files)

    assert check_s3_path_exists(
        cfg.scenario_builder.db_files
    ), f"Multinode caching only works in S3, but db_files path given was {cfg.scenario_builder.db_files}"

    all_files = get_db_filenames_from_load_path(cfg.scenario_builder.db_files)
    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info("Node ID: %s", str(node_id))
    file_chunks = chunk_list(all_files, num_nodes)
    current_chunk = file_chunks[node_id]
    current_chunk = list(current_chunk)
    logger.info("Current Chunk Length: %s", str(len(current_chunk)))

    return cast(List[str], current_chunk)


def scenario_based_split_and_extract_file_chunks_for_current_node(
    token_distribution: List[Tuple[str, str]], db_files_path: Path
) -> Tuple[List[str], List[str]]:
    """
    Splits the list of tuples (scenario_token, log_name) token_distribution into equal chunks and returns the
    list of the scenario tokens and the list of unique log files. The length of the two lists does not generally match.
    :param token_distribution: List[Tuple[scenario_token, log_name]] loaded from the csv files and concatenated
    :param db_files_path: path to the folder containing the log files
    :return: List of scenario tokens associated to the current machine (equally distributed over machines),
    and list of unique db files where these tokens are.
    """
    num_nodes = int(os.environ.get("NUM_NODES", 1))

    assert num_nodes > 1, (
        "The scenario-based split can only be called with num_nodes > 1, but was called"
        f"with num_nodes = {num_nodes}."
    )

    db_files_path_sanitized = sanitize_s3_path(db_files_path)
    assert check_s3_path_exists(
        db_files_path_sanitized
    ), f"Multinode caching only works in S3, but db_files path given was {db_files_path_sanitized}"

    token_distribution_chunk = chunk_list(token_distribution, num_nodes)
    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info("Node ID: %s", str(node_id))
    current_chunk = token_distribution_chunk[node_id]
    logger.info(f"This machine is about to process {len(current_chunk)} scenarios.")
    current_logs_chunk = list({os.path.join(db_files_path_sanitized, f"{pair[1]}.db") for pair in current_chunk})
    current_token_chunk = [pair[0] for pair in current_chunk]

    return current_token_chunk, current_logs_chunk


def get_all_scenarios_on_machine(
    log_db_files: List[str], worker: WorkerPool, cfg: DictConfig
) -> List[AbstractScenario]:
    """
    Gets all the scenarios from the chunk of log_db_files. It does it with parallelism on worker.
    :param log_db_files: names of the .db files
    :param worker: worker for parallelism
    :param cfg: config
    :return: the list of generated scenarios
    """
    OmegaConf.set_struct(cfg, False)
    cfg.scenario_builder.db_files = log_db_files
    OmegaConf.set_struct(cfg, True)

    logger.info("Found %s to process. Extracting scenarios and writing out the csv file.", str(len(log_db_files)))

    scenario_builder = build_scenario_builder(cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    scenarios: List[AbstractScenario] = scenario_builder.get_scenarios(scenario_filter, worker)
    return scenarios


def get_repartition_tokens(scenarios: List[AbstractScenario], cfg: DictConfig) -> Tuple[Any, Any]:
    """
    Generates all the scenario tokens that are a match with the requested features
    specified in the config files. It does that in a distributed manner, where each machine
    generates a file with the processed tokens, and at the end of each machine's generation,
    it reads in all the scenario tokens and gets its own chunk to process.
    In this way, we have an equal split over the scenarios across the machines.
    :param scenarios: list of scenarios generated by the current machine
    :param cfg: config
    :return: List of scenario tokens associated to the current machine (equally distributed over machines),
    and list of unique db files where these tokens are.
    """
    unique_job_id = get_unique_job_id()
    token_distribution_file_dir = sanitize_s3_path(
        Path(cfg.cache.cache_path).parent / Path("cache_token_distribution/cache") / unique_job_id
    )
    token_distribution_file_dir_barrier = sanitize_s3_path(
        Path(cfg.cache.cache_path).parent / Path("cache_token_distribution/barrier") / unique_job_id
    )

    _write_csv_file_to_s3(scenarios, token_distribution_file_dir)

    barrier = FileBackedBarrier(Path(token_distribution_file_dir_barrier))
    barrier.wait_barrier(
        activity_id="barrier_token_" + str(os.environ.get('NODE_RANK', 0)),
        expected_activity_ids={"barrier_token_" + str(el) for el in range(0, int(os.environ.get('NUM_NODES', 1)))},
        timeout_s=3600,
        poll_interval_s=0.5,
    )

    token_distribution = _get_all_generated_csv(token_distribution_file_dir)

    db_files_path = (
        Path(cfg.scenario_builder.db_files[0]).parent
        if isinstance(cfg.scenario_builder.db_files, (list, ListConfig))
        else Path(cfg.scenario_builder.db_files)
    )

    return scenario_based_split_and_extract_file_chunks_for_current_node(token_distribution, db_files_path)


def _write_csv_file_to_s3(scenarios: List[AbstractScenario], token_distribution_file_dir: str) -> None:
    """
    Used for the beginning of the caching job. The machine stores the scenario tokens and log names to a csv file.
    :param scenarios: list of scenarios to process to extract the tokens
    :param token_distribution_file_dir: path where to store the generated .csv file
    """
    token_distribution_file = os.path.join(token_distribution_file_dir, f"{os.environ.get('NODE_RANK', 0)}.csv")
    token_log_pairs = [(scenario.token, scenario.log_name) for scenario in scenarios]
    if not os.path.exists(token_distribution_file_dir):
        os.makedirs(token_distribution_file_dir)
    token_log_pairs_df = pd.DataFrame(token_log_pairs)
    # The following to_csv function handles both local and s3 paths.
    token_log_pairs_df.to_csv(token_distribution_file, index=False)
    logger.info(f"CSV file with {len(token_log_pairs)} scenarios written out to {token_distribution_file}.")


def _get_all_generated_csv(token_distribution_file_dir: str) -> List[Tuple[str, str]]:
    """
    Read the csv files that every machine in the cluster generated.
    :param token_distribution_file_dir: path where to the csv files are stored
    """
    token_distribution_file_list = [el for el in expand_s3_dir(token_distribution_file_dir) if el.endswith(".csv")]
    token_distribution_list = []
    bucket, file_path = split_s3_path(Path(token_distribution_file_list[0]))
    s3_store = S3Store(s3_prefix=os.path.join('s3://', bucket))
    for token_distribution_file in token_distribution_file_list:
        with s3_store.get(token_distribution_file) as f:
            token_distribution_list.append(pd.read_csv(f, delimiter=','))
    token_distribution_df = pd.concat(token_distribution_list, ignore_index=True)
    logger.info(f"CSV files read from {token_distribution_file_dir}.")

    token_distribution = token_distribution_df.values.tolist()
    logger.info(f"In total, {len(token_distribution)} scenarios have been found by all machines.")
    return cast(List[Tuple[str, str]], token_distribution)
