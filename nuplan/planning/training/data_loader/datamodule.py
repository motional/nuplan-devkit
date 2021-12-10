import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.utils.data
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.training.modeling.types import FeaturesType, move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_caching_preprocessor import FeatureCachingPreprocessor
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate

logger = logging.getLogger(__name__)


def create_dataset(
        samples: List[AbstractScenario],
        feature_and_targets_builders: FeatureCachingPreprocessor,
        dataset_fraction: float,
        dataset_name: str,
) -> torch.utils.data.Dataset:
    """
    Creates a dataset from a list of samples.

    :param samples: list of candidate samples
    :param feature_and_targets_builders: feature extractor
    :param dataset_fraction: fraction of the dataset to load
    :param dataset_name: val/test/train set name
    :return: the created torch dataset
    """
    # Sample the desired fraction from the total samples
    num_keep = int(len(samples) * dataset_fraction)
    selected_scenarios = random.sample(samples, num_keep)

    logger.info(f"Number of samples in {dataset_name} set: {len(selected_scenarios)}")
    return ScenarioDataset(
        scenarios=selected_scenarios,
        feature_caching_preprocessor=feature_and_targets_builders,
    )


class DataModule(pl.LightningDataModule):
    """
    Datamodule wrapping all preparation and dataset creation functionality.
    """

    def __init__(
            self,
            feature_and_targets_builders: FeatureCachingPreprocessor,
            splitter: AbstractSplitter,
            all_scenarios: List[AbstractScenario],
            train_fraction: float,
            val_fraction: float,
            test_fraction: float,
            dataloader_params: Dict[str, Any],
    ) -> None:
        """
        Initializes the class.

        :param feature_and_targets_builders: feature and targets builder used in dataset
        :param splitter: splitter object used to retrieve lists of samples to construct train/val/test sets
        :param train_fraction: fraction of training examples to load
        :param val_fraction: fraction of validation examples to load
        :param test_fraction: fraction of test examples to load
        :param dataloader_params: parameter dictionary passed to the dataloaders
        """
        super().__init__()

        assert train_fraction > 0.0, "Train fraction has to be larger than 0!"
        assert val_fraction > 0.0, "Validation fraction has to be larger than 0!"
        assert test_fraction >= 0.0, "Test fraction has to be larger/equal than 0!"

        # Datasets
        self._train_set: Optional[torch.utils.data.Dataset] = None
        self._val_set: Optional[torch.utils.data.Dataset] = None
        self._test_set: Optional[torch.utils.data.Dataset] = None

        # Feature computation
        self._feature_and_targets_builders = feature_and_targets_builders

        # Data splitter train/test/val
        self._splitter = splitter

        # Fractions
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction

        # Data loader for train/val/test
        self._dataloader_params = dataloader_params

        # Extract all samples
        self._all_samples = all_scenarios

    @property
    def feature_and_targets_builder(self) -> FeatureCachingPreprocessor:
        return self._feature_and_targets_builders

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the dataset for each target set depending on the training stage.

        This is called by every process in distributed training.

        :param stage: stage of training, can be "fit" or "test"
        """
        if stage is None:
            return

        if stage == 'fit':
            # Training Dataset
            train_samples = self._splitter.get_train_samples(self._all_samples)
            self._train_set = create_dataset(
                train_samples, self._feature_and_targets_builders, self._train_fraction, "train"
            )

            # Validation Dataset
            val_samples = self._splitter.get_val_samples(self._all_samples)
            self._val_set = create_dataset(
                val_samples, self._feature_and_targets_builders, self._val_fraction, "validation"
            )
        elif stage == 'test':
            # Testing Dataset
            test_samples = self._splitter.get_test_samples(self._all_samples)
            self._test_set = create_dataset(
                test_samples, self._feature_and_targets_builders, self._test_fraction, "test"
            )
        else:
            raise ValueError(f'Stage must be one of ["fit", "test"], got ${stage}.')

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Cleans up after a training stage.

        This is called by every process in distributed training.

        :param stage: stage of training, can be "fit" or "test"
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates the training dataloader.

        :raises RuntimeError: if this method is called without calling "setup()" first
        :return: the created torch dataloader
        """
        if self._train_set is None:
            raise RuntimeError('Data module has not been setup, call "setup()"')

        return torch.utils.data.DataLoader(
            dataset=self._train_set, **self._dataloader_params, shuffle=True, collate_fn=FeatureCollate()
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates the validation dataloader.

        :raises RuntimeError: if this method is called without calling "setup()" first
        :return: the created torch dataloader
        """
        if self._val_set is None:
            raise RuntimeError('Data module has not been setup, call "setup()"')

        return torch.utils.data.DataLoader(
            dataset=self._val_set, **self._dataloader_params, collate_fn=FeatureCollate()
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates the test dataloader.

        :raises RuntimeError: if this method is called without calling "setup()" first
        :return: the created torch dataloader
        """
        if self._test_set is None:
            raise RuntimeError('Data module has not been setup, call "setup()"')

        return torch.utils.data.DataLoader(
            dataset=self._test_set, **self._dataloader_params, collate_fn=FeatureCollate()
        )

    def transfer_batch_to_device(self, batch: Tuple[FeaturesType, ...], device: torch.device) \
            -> Tuple[FeaturesType, ...]:
        """
        Transfer batch to device
        :param batch: batch on origin device
        :param device: desired device
        :return: batch in new device
        """
        return tuple([move_features_type_to_device(features, device) for features in batch])
