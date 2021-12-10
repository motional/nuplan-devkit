from __future__ import annotations

import logging
import pathlib
from typing import Optional, Union

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder, \
    AbstractModelFeature
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCache

logger = logging.getLogger(__name__)


def compute_or_load_feature(scenario: AbstractScenario,
                            sample_dir: Optional[pathlib.Path],
                            builder: Union[AbstractFeatureBuilder, AbstractTargetBuilder],
                            storing_mechanism: FeatureCache) \
        -> AbstractModelFeature:
    """
    Compute features if non existent in cache, otherwise load them from cache
    :param scenario: for which features should be computed
    :param sample_dir: cache for this scenario, if None, we w
    :param builder: which builder should compute the features
    :param storing_mechanism: a way to store features
    :return features computed with builder
    """
    # Extract feature type
    feature_type = builder.get_feature_type()

    # Construct Folders and Filenames
    file_name = sample_dir / pathlib.Path(builder.get_feature_unique_name()) if sample_dir else None

    if file_name and storing_mechanism.exists_feature_cache(file_name):
        # In case folder exists, load it
        logger.debug(f"Loading feature: {file_name} from a file!")

        feature = storing_mechanism.load_computed_feature_from_folder(file_name, feature_type)
    else:
        # If cached file doesnt exists, compute and store features

        if isinstance(builder, AbstractFeatureBuilder):
            feature = builder.get_features_from_scenario(scenario)
        elif isinstance(builder, AbstractTargetBuilder):
            feature = builder.get_targets(scenario)
        else:
            raise ValueError(f"Unknown builder type: {type(builder)}")

        # If desired folder exists, save it
        if file_name:
            # In case feature was not computed, compute it and store it
            logger.debug(f"Saving feature: {file_name} to a file!")
            # Load feature from a directory
            file_name.parent.mkdir(parents=True, exist_ok=True)
            # Store dataclass into a file
            storing_mechanism.store_computed_feature_to_folder(file_name, feature)

    return feature
