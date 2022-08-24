import gc
import itertools
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

from omegaconf import DictConfig, OmegaConf

from nuplan.common.utils.s3_utils import check_s3_path_exists
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
    save_cache_metadata,
)
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.utils.multithreading.worker_utils import chunk_list, worker_map

logger = logging.getLogger(__name__)


def cache_scenarios_oneshot(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "db_file": the db file to process
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """
    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_oneshot_internal(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())
        db_files: List[str] = [cast(str, a["db_file"]) for a in args]
        cfg: DictConfig = args[0]["cfg"]

        logger.info(
            "Starting worker thread with thread_id=%s, node_id=%s to process db files %s",
            thread_id,
            node_id,
            db_files,
        )
        OmegaConf.set_struct(cfg, False)
        cfg.scenario_builder.db_files = db_files
        OmegaConf.set_struct(cfg, True)

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

        scenario_builder = build_scenario_builder(cfg)
        scenario_filter = build_scenario_filter(cfg.scenario_filter)
        seq_worker = Sequential()
        scenarios: List[AbstractScenario] = scenario_builder.get_scenarios(scenario_filter, seq_worker)

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

    result = cache_scenarios_oneshot_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result


def cache_scenarios_parallel_oneshot(
    current_chunk: List[str], cfg: DictConfig, worker: WorkerPool
) -> List[CacheMetadataEntry]:
    """
    Perform the scenario caching in "one shot".
    That is, read in the scenario and perform feature computation in a single function.

    This is done in one function to save memory when using ray workers - serializing the DictConfig uses much less memory
        than serializing AbstractScenarios.
    :param current_chunk: The chunk of files to process.
    :param cfg: The configuration to use for the feature builders and scenario builders.
    :param worker: The worker pool to user for parallelization.
    :return: File_names for each of the valid cached features and targets
    """
    data_points = [{"db_file": cc, "cfg": cfg} for cc in current_chunk]

    logger.info("Starting dataset caching of %s files...", str(len(data_points)))

    cache_results = worker_map(worker, cache_scenarios_oneshot, data_points)
    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    logger.info("Completed dataset caching! Failed features and targets: %s out of %s", str(num_fail), str(num_total))

    valid_cache_metadata_entries = [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]
    return valid_cache_metadata_entries


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

    logger.info("Found %s to process.", str(len(log_db_files)))

    cached_metadata = cache_scenarios_parallel_oneshot(log_db_files, cfg, worker)

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
    ), "Multinode caching only works in S3, but db_files path given was {cfg.scenario_builder.db_files}"

    all_files = get_db_filenames_from_load_path(cfg.scenario_builder.db_files)
    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info("Node ID: %s", str(node_id))
    file_chunks = chunk_list(all_files, num_nodes)
    current_chunk = file_chunks[node_id]
    current_chunk = list(current_chunk)
    logger.info("Current Chunk Length: %s", str(len(current_chunk)))

    return cast(List[str], current_chunk)
