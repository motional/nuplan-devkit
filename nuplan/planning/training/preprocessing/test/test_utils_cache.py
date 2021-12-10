import logging
import pathlib
import time
import unittest

import numpy as np
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCache, FeatureCachePickle

logger = logging.getLogger(__name__)


class TestUtilsCache(unittest.TestCase):

    def setUp(self) -> None:
        self.caching_engine = [FeatureCachePickle()]

    def test_storing_to_cache_vector_map(self) -> None:
        """
        Test storing feature to cache
        """
        folder = pathlib.Path("/tmp/feature")
        large_size = int(50)
        feature = VectorMap(coords=[np.zeros((large_size, 2, 2)).astype(np.float32)],
                            multi_scale_connections=[{1: np.zeros((large_size, 2)).astype(np.float32)}])
        folder = folder / "VectorMap"

        for cacher in self.caching_engine:
            folder.parent.mkdir(parents=True, exist_ok=True)
            time_now = time.time()
            loaded_feature: VectorMap = self.store_and_load(cacher, folder, feature)
            time_later = time.time()
            logger.debug(f"Cache: {type(cacher)} = {time_later - time_now}")

            self.assertEqual(feature.num_of_batches, loaded_feature.num_of_batches)
            self.assertEqual(1, loaded_feature.num_of_batches)
            self.assertEqual(feature.coords[0].shape, loaded_feature.coords[0].shape)
            self.assertEqual(feature.multi_scale_connections[0][1].shape,
                             loaded_feature.multi_scale_connections[0][1].shape)

    def test_storing_to_cache_raster(self) -> None:
        """
        Test storing feature to cache
        """
        folder = pathlib.Path("/tmp/feature")
        feature = Raster(data=np.zeros((244, 244, 3)))
        folder = folder / "Raster"

        for cacher in self.caching_engine:
            folder.parent.mkdir(parents=True, exist_ok=True)
            loaded_feature = self.store_and_load(cacher, folder, feature)
            self.assertEqual(feature.data.shape, loaded_feature.data.shape)

    def store_and_load(self, cacher: FeatureCache, folder: pathlib.Path, feature: AbstractModelFeature) \
            -> AbstractModelFeature:
        time_now = time.time()
        cacher.store_computed_feature_to_folder(folder, feature)
        logger.debug(f"store_computed_feature_to_folder: {type(cacher)} = {time.time() - time_now}")
        time_now = time.time()
        out = cacher.load_computed_feature_from_folder(folder, feature)
        logger.debug(f"load_computed_feature_from_folder: {type(cacher)} = {time.time() - time_now}")
        self.assertIsInstance(out, type(feature))
        return out


if __name__ == '__main__':
    unittest.main()
