import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Type, cast

from hydra._internal.utils import _locate
from omegaconf import DictConfig

from nuplan.common.utils.s3_utils import check_s3_path_exists, expand_s3_dir
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map

logger = logging.getLogger(__name__)


def get_s3_scenario_cache(cache_path: str, feature_names: Set[str]) -> List[Path]:
    """
    Get a list of cached scenario paths from a remote (S3) cache.
    :param cache_path: Root path of the remote cache dir.
    :param feature_names: Set of required feature names to check when loading scenario paths from the cache.
    :return: List of discovered cached scenario paths.
    """
    # Retrieve all filenames contained in the remote location.
    assert check_s3_path_exists(cache_path), 'Remote cache {cache_path} does not exist!'
    s3_filenames = expand_s3_dir(cache_path)
    assert len(s3_filenames) > 0, 'No files found in the remote cache {cache_path}!'

    # Create a 2-level hash with log names and scenario tokens as keys and the set of contained features as values.
    cache_map: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for s3_filename in s3_filenames:
        path = Path(s3_filename)
        cache_map[path.parent.parent.name][path.parent.name].add(path.stem)

    # Keep only dir paths that contain all required feature names
    scenario_cache_paths = [
        Path(f'{log_name}/{scenario_token}')
        for log_name, scenarios in cache_map.items()
        for scenario_token, features in scenarios.items()
        if not (feature_names - features)
    ]

    return scenario_cache_paths


def get_local_scenario_cache(cache_path: str, feature_names: Set[str]) -> List[Path]:
    """
    Get a list of cached scenario paths from a local cache.
    :param cache_path: Root path of the local cache dir.
    :param feature_names: Set of required feature names to check when loading scenario paths from the cache.
    :return: List of discovered cached scenario paths.
    """
    cache_dir = Path(cache_path)
    assert cache_dir.exists(), f'Local cache {cache_dir} does not exist!'
    assert any(cache_dir.iterdir()), f'No files found in the local cache {cache_dir}!'

    candidate_scenario_dirs = [path for log_dir in cache_dir.iterdir() for path in log_dir.iterdir()]

    # Keep only dir paths that contains all required feature names
    scenario_cache_paths = [
        path
        for path in candidate_scenario_dirs
        if not (feature_names - {feature_name.stem for feature_name in path.iterdir()})
    ]

    return scenario_cache_paths


def extract_scenarios_from_cache(
    cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper
) -> List[AbstractScenario]:
    """
    Build the scenario objects that comprise the training dataset from cache.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :param model: NN model used for training.
    :return: List of extracted scenarios.
    """
    cache_path = str(cfg.cache.cache_path)

    # Find all required feature/target names to load from cache
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()
    feature_names = {builder.get_feature_unique_name() for builder in feature_builders + target_builders}

    # Get cached scenario paths locally or remotely
    scenario_cache_paths = (
        get_s3_scenario_cache(cache_path, feature_names)
        if cache_path.startswith('s3://')
        else get_local_scenario_cache(cache_path, feature_names)
    )

    # Get the scenario class and default arguments to use when instantiating a scenario.
    scenario_builder_cls: Type[AbstractScenarioBuilder] = _locate(cfg.scenario_builder._target_)
    scenario_cls = scenario_builder_cls.get_scenario_type()
    scenario_args = {
        'db': None,
        'scenario_type': None,
        'scenario_extraction_info': None,
        'ego_vehicle_parameters': None,
    }

    def create_scenario_from_paths(paths: List[Path]) -> List[AbstractScenario]:
        """
        Create scenario objects from a list of cache paths in the format of ".../log_name/scenario_token".
        :param paths: List of paths to load scenarios from.
        :return: List of created scenarios.
        """
        scenarios = [
            scenario_cls(log_name=path.parent.name, initial_lidar_token=path.name, **scenario_args) for path in paths
        ]

        return scenarios

    scenarios = worker_map(worker, create_scenario_from_paths, scenario_cache_paths)

    return cast(List[AbstractScenario], scenarios)


def extract_scenarios_from_dataset(cfg: DictConfig, worker: WorkerPool) -> List[AbstractScenario]:
    """
    Extract and filter scenarios by loading a dataset using the scenario builder.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: List of extracted scenarios.
    """
    scenario_builder = build_scenario_builder(cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    scenarios: List[AbstractScenario] = scenario_builder.get_scenarios(scenario_filter, worker)

    return scenarios


def build_scenarios(cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper) -> List[AbstractScenario]:
    """
    Build the scenario objects that comprise the training dataset.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :param model: NN model used for training.
    :return: List of extracted scenarios.
    """
    scenarios = (
        extract_scenarios_from_cache(cfg, worker, model)
        if cfg.cache.use_cache_without_dataset
        else extract_scenarios_from_dataset(cfg, worker)
    )

    logger.info(f'Extracted {len(scenarios)} scenarios for training')
    assert len(scenarios) > 0, 'No scenarios were retrieved for training, check the scenario_filter parameters!'

    return scenarios
