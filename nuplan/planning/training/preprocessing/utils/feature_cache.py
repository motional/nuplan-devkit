from __future__ import annotations

import abc
import gzip
import os
import pathlib
import pickle
from io import BytesIO
from typing import Type, cast

import joblib

from nuplan.common.utils.s3_utils import check_s3_path_exists
from nuplan.database.common.blob_store.s3_store import S3Store
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature


class FeatureCache(abc.ABC):
    """
    Cache and load features to a file
    """

    @abc.abstractclassmethod
    def exists_feature_cache(self, feature_file: pathlib.Path) -> bool:
        """
        :return true in case feature file exists
        """
        pass

    @abc.abstractmethod
    def with_extension(self, feature_file: pathlib.Path) -> str:
        """
        Append extension
        :param feature_file: input feature file name
        :return filename with extension
        """
        pass

    @abc.abstractmethod
    def store_computed_feature_to_folder(self, feature_file: pathlib.Path, feature: AbstractModelFeature) -> bool:
        """
        Store computed features into folder
        As of now, feature types we support are only np.ndarray and dict
        :param feature_file: where features should be stored
        :param feature: feature types
        """
        pass

    @abc.abstractmethod
    def load_computed_feature_from_folder(
        self, feature_file: pathlib.Path, feature_type: Type[AbstractModelFeature]
    ) -> AbstractModelFeature:
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

    def exists_feature_cache(self, feature_file: pathlib.Path) -> bool:
        """Inherited, see superclass."""
        return pathlib.Path(self.with_extension(feature_file)).exists()

    def with_extension(self, feature_file: pathlib.Path) -> str:
        """Inherited, see superclass."""
        return str(feature_file.with_suffix(".gz"))

    def store_computed_feature_to_folder(self, feature_file: pathlib.Path, feature: AbstractModelFeature) -> bool:
        """Inherited, see superclass."""
        serializable_dict = feature.serialize()
        # Use compresslevel = 1 to compress the size but also has fast write and read.
        with gzip.open(self.with_extension(feature_file), 'wb', compresslevel=1) as f:
            pickle.dump(serializable_dict, f)
        return True

    def load_computed_feature_from_folder(
        self, feature_file: pathlib.Path, feature_type: Type[AbstractModelFeature]
    ) -> AbstractModelFeature:
        """Inherited, see superclass."""
        with gzip.open(self.with_extension(feature_file), 'rb') as f:
            data = pickle.load(f)
        return feature_type.deserialize(data)


class FeatureCacheS3(FeatureCache):
    """
    Store features remotely in S3
    """

    def __init__(self, s3_path: str) -> None:
        """
        Initialize the S3 remote feature cache.
        :param s3_path: Path to S3 directory where features will be stored to or loaded from.
        """
        self._store = S3Store(s3_path, show_progress=False)  # TODO: Utilize passed path in S3Store

    def exists_feature_cache(self, feature_file: pathlib.Path) -> bool:
        """Inherited, see superclass."""
        return cast(bool, check_s3_path_exists(self.with_extension(feature_file)))

    def with_extension(self, feature_file: pathlib.Path) -> str:
        """Inherited, see superclass."""
        fixed_s3_filename = f's3://{str(feature_file).lstrip("s3:/")}'
        return f'{fixed_s3_filename}.bin'

    def store_computed_feature_to_folder(self, feature_file: pathlib.Path, feature: AbstractModelFeature) -> bool:
        """Inherited, see superclass."""
        # Serialize feature object to bytes
        serialized_feature = BytesIO()
        joblib.dump(feature, serialized_feature)

        # Set serialized file reference point to beginning of the file
        serialized_feature.seek(os.SEEK_SET)

        # Put serialized feature in the remote feature store
        storage_key = self.with_extension(feature_file)

        successfully_stored_feature = self._store.put(storage_key, serialized_feature, ignore_if_client_error=True)

        return cast(bool, successfully_stored_feature)

    def load_computed_feature_from_folder(
        self, feature_file: pathlib.Path, feature_type: Type[AbstractModelFeature]
    ) -> AbstractModelFeature:
        """Inherited, see superclass."""
        # Retrieve serialized feature from the remote feature store
        storage_key = self.with_extension(feature_file)
        serialized_feature = self._store.get(storage_key)

        # Deserialize feature object from bytes
        feature = joblib.load(serialized_feature)

        return feature
