from __future__ import annotations

import abc
import pathlib
import pickle
from typing import Type

from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature


class FeatureCache(abc.ABC):
    """
    Cache and load features to a file
    """

    def exists_feature_cache(self, feature_file: pathlib.Path) -> bool:
        """
        :return true in case feature file exists
        """
        return self.with_extension(feature_file).exists()

    @abc.abstractmethod
    def with_extension(self, feature_file: pathlib.Path) -> pathlib.Path:
        """
        Append extension
        :param feature_file: input feature file name
        :return filename with extension
        """
        pass

    @abc.abstractmethod
    def store_computed_feature_to_folder(self, feature_file: pathlib.Path, feature: AbstractModelFeature) -> None:
        """
        Store computed features into folder
        As of now, feature types we support are only np.ndarray and dict
        :param feature_file: where features should be stored
        :param feature: feature types
        """
        pass

    @abc.abstractmethod
    def load_computed_feature_from_folder(self, feature_file: pathlib.Path, feature_type: Type[AbstractModelFeature]) \
            -> AbstractModelFeature:
        """
        Load feature of type from a folder
        :param feature_file: where all files should be located
        :param feature_type: type of feature to be loaded
        :return: loaded feature
        """
        pass


class FeatureCachePickle(FeatureCache):
    """
    Store features with pickle
    """

    def with_extension(self, feature_file: pathlib.Path) -> pathlib.Path:
        """ Inherited, see superclass. """
        return feature_file.with_suffix(".pkl")

    def store_computed_feature_to_folder(self, feature_file: pathlib.Path, feature: AbstractModelFeature) -> None:
        """ Inherited, see superclass. """
        serializable_dict = feature.serialize()
        with open(self.with_extension(feature_file), 'wb') as f:
            pickle.dump(serializable_dict, f)

    def load_computed_feature_from_folder(self, feature_file: pathlib.Path, feature_type: Type[AbstractModelFeature]) \
            -> AbstractModelFeature:
        """ Inherited, see superclass. """
        with open(self.with_extension(feature_file), 'rb') as f:
            data = pickle.load(f)
        return feature_type.deserialize(data)
