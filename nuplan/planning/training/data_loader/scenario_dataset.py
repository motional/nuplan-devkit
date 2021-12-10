import logging
from typing import List, Tuple

import torch.utils.data
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_caching_preprocessor import FeatureCachingPreprocessor

logger = logging.getLogger(__name__)


class ScenarioDataset(torch.utils.data.Dataset):
    """
    Dataset responsible for consuming scenarios and producing pairs of model inputs/outputs.
    """

    def __init__(self,
                 scenarios: List[AbstractScenario],
                 feature_caching_preprocessor: FeatureCachingPreprocessor) -> None:
        """
        Initializes the class.
        :param scenarios: list of scenarios to use as dataset examples
        :param feature_caching_preprocessor: feature and targets builder that converts samples to model features
        """
        super().__init__()
        if len(scenarios) == 0:
            logger.warning('The dataset has no samples')

        self._scenarios = scenarios
        self._computator = feature_caching_preprocessor

    def __getitem__(self, idx: int) -> Tuple[FeaturesType, TargetsType]:
        """
        Retrieves the dataset examples corresponding to the input index

        :param idx: input index
        :return: model features and targets
        """
        scenario = self._scenarios[idx]

        features, targets = self._computator.compute_features(scenario)
        features = {key: value.to_feature_tensor() for key, value in features.items()}
        targets = {key: value.to_feature_tensor() for key, value in targets.items()}

        return features, targets

    def __len__(self) -> int:
        """
        Returns the size of the dataset (number of samples)

        :return: size of dataset
        """
        return len(self._scenarios)
