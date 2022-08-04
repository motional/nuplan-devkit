from __future__ import annotations

import logging
import pathlib
from typing import Optional, Union

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCache

logger = logging.getLogger(__name__)


def compute_or_load_feature(
    scenario: AbstractScenario,
    cache_path: Optional[pathlib.Path],
    builder: Union[AbstractFeatureBuilder, AbstractTargetBuilder],
    storing_mechanism: FeatureCache,
    force_feature_computation: bool,
) -> AbstractModelFeature:
    """
    Compute features if non existent in cache, otherwise load them from cache
    :param scenario: for which features should be computed
    :param cache_path: location of cached features
    :param builder: which builder should compute the features
    :param storing_mechanism: a way to store features
    :param force_feature_computation: if true, even if cache exists, it will be overwritten
    :return features computed with builder
    """
    cache_path_available = cache_path is not None

    # Filename of the cached features/targets
    file_name = (
        cache_path / scenario.log_name / scenario.scenario_type / scenario.token / builder.get_feature_unique_name()
        if cache_path_available
        else None
    )

    # If feature recomputation is desired or cached file doesnt exists, compute the feature
    if force_feature_computation or not cache_path_available or not storing_mechanism.exists_feature_cache(file_name):
        logger.debug("Computing feature...")
        if isinstance(builder, AbstractFeatureBuilder):
            feature = builder.get_features_from_scenario(scenario)
        elif isinstance(builder, AbstractTargetBuilder):
            feature = builder.get_targets(scenario)
        else:
            raise ValueError(f"Unknown builder type: {type(builder)}")

        # If caching is enabled, store the feature
        if feature.is_valid and cache_path_available:
            logger.debug(f"Saving feature: {file_name} to a file...")
            file_name.parent.mkdir(parents=True, exist_ok=True)
            storing_mechanism.store_computed_feature_to_folder(file_name, feature)
    else:
        # In case the feature exists in the cache, load it
        logger.debug(f"Loading feature: {file_name} from a file...")
        feature = storing_mechanism.load_computed_feature_from_folder(file_name, builder.get_feature_type())
        assert feature.is_valid, 'Invalid feature loaded from cache!'

    return feature
