import logging
import pathlib
import time
import unittest

import numpy as np

from nuplan.database.common.blob_store.test.mock_s3_store import MockS3Store
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCache, FeatureCachePickle, FeatureCacheS3

logger = logging.getLogger(__name__)


class TestUtilsCache(unittest.TestCase):
    """Test caching utilities."""

    def setUp(self) -> None:
        """Set up test case."""
        local_cache_path = '/tmp/cache'
        s3_cache_path = 's3://tmp/cache'
        self.cache_paths = [local_cache_path, s3_cache_path]

        local_store = FeatureCachePickle()
        s3_store = FeatureCacheS3(s3_cache_path)
        s3_store._store = MockS3Store()
        self.cache_engines = [local_store, s3_store]

    def test_storing_to_cache_vector_map(self) -> None:
        """
        Test storing feature to cache
        """
        dim = 50
        feature = VectorMap(
            coords=[np.zeros((dim, 2, 2)).astype(np.float32)],
            lane_groupings=[[np.zeros(dim).astype(np.float32)]],
            multi_scale_connections=[{1: np.zeros((dim, 2)).astype(np.float32)}],
            on_route_status=[np.zeros((dim, 2)).astype(np.float32)],
            traffic_light_data=[np.zeros((dim, 4)).astype(np.float32)],
        )

        for cache_path, cache in zip(self.cache_paths, self.cache_engines):
            folder = pathlib.Path(cache_path) / 'tmp_log_name' / 'tmp_scenario_token' / 'vector_map'

            if not str(folder).startswith('s3:/'):
                folder.parent.mkdir(parents=True, exist_ok=True)

            time_now = time.time()
            loaded_feature: VectorMap = self.store_and_load(cache, folder, feature)
            time_later = time.time()
            logger.debug(f"Cache: {type(cache)} = {time_later - time_now}")

            self.assertEqual(feature.num_of_batches, loaded_feature.num_of_batches)
            self.assertEqual(1, loaded_feature.num_of_batches)
            self.assertEqual(feature.coords[0].shape, loaded_feature.coords[0].shape)
            self.assertEqual(feature.lane_groupings[0][0].shape, loaded_feature.lane_groupings[0][0].shape)
            self.assertEqual(
                feature.multi_scale_connections[0][1].shape, loaded_feature.multi_scale_connections[0][1].shape
            )

    def test_storing_to_cache_raster(self) -> None:
        """
        Test storing feature to cache
        """
        feature = Raster(data=np.zeros((244, 244, 3)))

        for cache_path, cache in zip(self.cache_paths, self.cache_engines):
            folder = pathlib.Path(cache_path) / 'tmp_log_name' / 'tmp_scenario_token' / 'raster'

            if not str(folder).startswith('s3:/'):
                folder.parent.mkdir(parents=True, exist_ok=True)

            loaded_feature = self.store_and_load(cache, folder, feature)
            self.assertEqual(feature.data.shape, loaded_feature.data.shape)

    def store_and_load(
        self, cache: FeatureCache, folder: pathlib.Path, feature: AbstractModelFeature
    ) -> AbstractModelFeature:
        """
        Store feature and load it back.
        :param cache: Caching mechanism to use.
        :param folder: Folder to store feature.
        :param feature: Feature to store.
        :return: Loaded feature.
        """
        time_now = time.time()
        cache.store_computed_feature_to_folder(folder, feature)
        logger.debug(f"store_computed_feature_to_folder: {type(cache)} = {time.time() - time_now}")

        time_now = time.time()
        out = cache.load_computed_feature_from_folder(folder, feature)
        logger.debug(f"load_computed_feature_from_folder: {type(cache)} = {time.time() - time_now}")

        self.assertIsInstance(out, type(feature))

        return out


if __name__ == '__main__':
    unittest.main()
