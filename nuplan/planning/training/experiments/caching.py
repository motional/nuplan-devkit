import logging
from typing import List

from omegaconf import DictConfig

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.scenario_builder import extract_scenarios_from_dataset
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

logger = logging.getLogger(__name__)


def cache_scenarios_parallel(
    scenarios: List[AbstractScenario],
    preprocessor: FeaturePreprocessor,
    worker: WorkerPool,
) -> int:
    """
    Cache all scenarios through the preprocessor in multiple processes.
    :param scenarios: List of scenarios to cache.
    :param preprocessor: Preprocessor object used for computing and caching features/targets given a scenario.
    :param worker: Worker to use for parallel processing.
    :return: Number of scenarios that failed the preprocessing.
    """

    def cache_scenarios(scenarios: List[AbstractScenario]) -> List[int]:
        """
        Helper function to cache a chunk of scenarios sequentially.
        :param scenarios: List of scenarios to cache.
        :return: Number of scenarios that failed the preprocessing.
        """
        num_failures = 0

        for scenario in scenarios:
            features, targets = preprocessor.compute_features(scenario)
            num_failures += any(not feature.is_valid for feature in list(features.values()) + list(targets.values()))

        return [num_failures]

    num_failed_samples_per_worker = worker_map(worker, cache_scenarios, scenarios)
    num_failed_samples = sum(num_failed_samples_per_worker)

    return num_failed_samples


def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    # Build model to get required feature/target builders
    model = build_torch_module_wrapper(cfg.model)
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()
    del model

    # Create feature preprocessor
    assert cfg.cache.cache_path is not None, f'Cache path cannot be None when caching, got {cfg.cache.cache_path}'
    feature_preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )

    # Extract scenarios to be cached from dataset
    scenarios = extract_scenarios_from_dataset(cfg, worker)

    # Cache scenarios in parallel
    logger.info('Starting dataset caching...')
    num_failed_samples = cache_scenarios_parallel(scenarios, feature_preprocessor, worker)
    logger.info(f'Completed dataset caching! Failed samples: {num_failed_samples} out of {len(scenarios)}.')
