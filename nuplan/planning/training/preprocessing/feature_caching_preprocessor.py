from __future__ import annotations

import logging
import pathlib
import traceback
from typing import List, Optional, Tuple, Type, Union

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder, \
    AbstractModelFeature
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from nuplan.planning.training.preprocessing.utils.utils_cache import compute_or_load_feature

logger = logging.getLogger(__name__)


class FeatureCachingPreprocessor:
    """
    Compute features and targets for a scenario. This class also manages cache. If a feature/target
    is not present in a cache, it is computed, otherwise it is loaded
    """

    def __init__(self,
                 cache_dir: Optional[Union[str, pathlib.Path]],
                 force_feature_computation: bool,
                 feature_builders: List[AbstractFeatureBuilder],
                 target_builders: List[AbstractTargetBuilder]):
        """
        Create feature computator
        :param cache_dir: whether to cache features
        :param force_feature_computation: if true, even if cache exists, it will be overwritten
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        """
        self._cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        self._force_feature_computation = force_feature_computation
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._storing_mechanism = FeatureCachePickle()
        assert len(feature_builders) != 0, "Number of feature builders has to be grater than 0!"

    @property
    def feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: all feature builders
        """
        return self._feature_builders

    @property
    def target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: all target builders
        """
        return self._target_builders

    def get_list_of_feature_types(self) -> List[Type[AbstractModelFeature]]:
        """
        :return all features that are computed by the builders
        """
        return [builder.get_feature_type() for builder in self._feature_builders]

    def get_list_of_target_types(self) -> List[Type[AbstractModelFeature]]:
        """
        :return all targets that are computed by the builders
        """
        return [builder.get_feature_type() for builder in self._target_builders]

    def compute_features(self, scenario: AbstractScenario) -> Tuple[FeaturesType, TargetsType]:
        """
        Compute features for a scenario, in case cache_dir is set, features will be stored in cache,
        otherwise just recomputed
        :param scenario for which features and targets should be computed
        :return: model features and targets
        """
        try:
            sample_token = scenario.token

            # Whether to force computation
            force_feature_computation = self._force_feature_computation or not self._cache_dir

            # Location of the cached features/targets, of None if we want to recompute features
            sample_cache_dir = self._cache_dir / sample_token if not force_feature_computation else None

            # Features
            all_features: FeaturesType = self._compute_all_features(scenario, sample_cache_dir, self._feature_builders)

            # Targets
            all_targets: TargetsType = self._compute_all_features(scenario, sample_cache_dir, self._target_builders)

            return all_features, all_targets
        except Exception as error:
            msg = f"Failed to compute features for scenario token {scenario.token}\nError: {error}"
            logger.error(msg)
            traceback.print_exc()
            raise RuntimeError(msg)

    def _compute_all_features(self, scenario: AbstractScenario,
                              sample_cache_dir: Optional[pathlib.Path],
                              builders: List[Union[AbstractFeatureBuilder, AbstractTargetBuilder]]) \
            -> Union[FeaturesType, TargetsType]:
        """
        Compute all features/targets from builders for scenario
        :param scenario: for which features should be computed
        :param sample_cache_dir: where cache should be stored
        :param builders: to use for feature computation
        :return: computed features/targets
        """
        # Features
        all_features: FeaturesType = {}
        for builder in builders:
            feature = compute_or_load_feature(scenario, sample_cache_dir, builder, self._storing_mechanism)
            all_features[builder.get_feature_unique_name()] = feature
        return all_features
