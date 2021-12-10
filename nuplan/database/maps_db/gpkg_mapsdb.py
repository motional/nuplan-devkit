import fcntl
import glob
import gzip
import json
import logging
import os
import time
import warnings
from functools import lru_cache
from mmap import PROT_READ, mmap
from tempfile import TemporaryDirectory
from typing import List, Sequence

import fiona
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import rasterio
from nuplan.database.common.blob_store.creator import BlobStoreCreator
from nuplan.database.maps_db import layer_dataset_ops
from nuplan.database.maps_db.imapsdb import IMapsDB
from nuplan.database.maps_db.layer import MapLayer
from nuplan.database.maps_db.metadata import MapLayerMeta

logger = logging.getLogger(__name__)

# To silence NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class GPKGMapsDBException(Exception):
    def __init__(self, message: str) -> None:
        """
        Constructor.
        :param message: Exception message.
        """
        super().__init__(message)


class GPKGMapsDB(IMapsDB):
    """ GPKG MapsDB implementation. """

    def __init__(self, map_version: str, map_root: str) -> None:
        """
        Constructor.
        :param map_version: Version of map.
        :param map_root: Root folder of the maps.
        """
        self._map_version = map_version
        self._map_root = map_root

        self._blob_store = BlobStoreCreator.create_mapsdb(map_root=self._map_root)
        version_file = self._blob_store.get(f"{self._map_version}.json")
        self._metadata = json.load(version_file)
        # The dimensions of the maps are hard-coded for the 4 locations.
        self.maps_dimension = {'sg-one-north': (21070, 28060),
                               'us-ma-boston': (20380, 28730),
                               'us-nv-las-vegas-strip': (69820, 30120),
                               'us-pa-pittsburgh-hazelwood': (22760, 23090)}

    # Metadata accessors ######################################################

    @property
    def version_names(self) -> List[str]:
        """
        Lists the map version names for all valid map locations, e.g.
        ['9.17.1964', '9.4.1630', '9.15.1915', '9.17.1937']
        """
        return [self._metadata[location]["version"] for location in self.get_locations()]

    def get_map_version(self) -> str:
        """ Inherited, see superclass. """
        return self._map_version

    def get_version(self, location: str) -> str:
        """ Inherited, see superclass. """
        return str(self._metadata[location]["version"])

    def _get_shape(self, location: str, layer_name: str) -> List[int]:
        """
        Gets the shape of a layer given the map location and layer name.
        :param location: Name of map location, e.g. "sg-one-north". See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        if layer_name == 'intensity':
            return self._metadata[location]["layers"][layer_name]["shape"]  # type: ignore
        else:
            # The dimensions of other map layers are using the hard-coded values.
            return list(self.maps_dimension[location])

    def _get_transform_matrix(self, location: str, layer_name: str) -> npt.NDArray[np.float64]:
        """
        Get transformation matrix of a layer given location and layer name.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """

        return np.array(self._metadata[location]["layers"][layer_name]["transform_matrix"])

    @staticmethod
    def is_binary(layer_name: str) -> bool:
        """
        Checks if the layer is binary.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        return layer_name in ["drivable_area", "intersection", "pedestrian_crossing", "walkway", "walk_way"]

    @staticmethod
    def _can_dilate(layer_name: str) -> bool:
        """
        If the layer can be dilated.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        return layer_name in ["drivable_area"]

    # public interface #######################################################

    def get_locations(self) -> Sequence[str]:
        """
        Gets the list of available location in this GPKGMapsDB version.
        """
        return self._metadata.keys()  # type: ignore

    def layer_names(self, location: str) -> Sequence[str]:
        """ Inherited, see superclass. """
        gpkg_layers = self._metadata[location]["layers"].keys()
        return list(filter(lambda x: '_distance_px' not in x, gpkg_layers))

    def load_layer(self, location: str, layer_name: str) -> MapLayer:
        """ Inherited, see superclass. """

        is_bin = self.is_binary(layer_name)
        can_dilate = self._can_dilate(layer_name)
        layer_data = self._get_layer_matrix(location, layer_name)
        transform_matrix = self._get_transform_matrix(location, layer_name)

        # We assume that the map's pixel-per-meter ratio is the same in the x and y directions,
        # since the MapLayer class requires that.
        precision = 1 / transform_matrix[0, 0]
        layer_meta = MapLayerMeta(name=layer_name,
                                  md5_hash="not_used_for_gpkg_mapsdb",
                                  can_dilate=can_dilate,
                                  is_binary=is_bin,
                                  precision=precision)

        distance_matrix = None
        # if can_dilate:
        #     distance_matrix = self._get_distance_matrix(location, layer_name)

        return MapLayer(data=layer_data,
                        metadata=layer_meta,
                        joint_distance=distance_matrix,
                        transform_matrix=transform_matrix)

    @lru_cache(maxsize=None)
    def load_vector_layer(self, location: str, layer_name: str) -> gpd.geodataframe:
        """ Inherited, see superclass. """

        # TODO: Remove temporary workaround once map_version is cleaned
        location = location.replace('.gpkg', '')

        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        self._blob_store.save_to_disk(rel_path)

        with warnings.catch_warnings():
            # Suppress the warnings from the GPKG operations below so that they don't spam the training logs.
            warnings.filterwarnings("ignore")

            # The projected coordinate system depends on which UTM zone the mapped location is in.
            map_meta = gpd.read_file(path_on_disk, layer="meta")
            projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]

            gdf_in_pixel_coords = gpd.read_file(path_on_disk, layer=layer_name)
            gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)

            # Restore "fid" column, which isn't included by default due to a quirk.
            # See http://kuanbutts.com/2019/07/02/gpkg-write-from-geopandas/
            with fiona.open(path_on_disk, layer=layer_name) as fiona_collection:
                gdf_in_utm_coords["fid"] = [f["id"] for f in fiona_collection]

        return gdf_in_utm_coords

    def vector_layer_names(self, location: str) -> Sequence[str]:
        """ Inherited, see superclass. """

        # TODO: Remove temporary workaround once map_version is cleaned
        location = location.replace('.gpkg', '')

        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        self._blob_store.save_to_disk(rel_path)

        return fiona.listlayers(path_on_disk)  # type: ignore

    def purge_cache(self) -> None:
        """ Inherited, see superclass. """

        logger.debug("Purging cache...")
        for f in glob.glob(os.path.join(self._map_root, "gpkg", "*")):
            os.remove(f)
        logger.debug("Done purging cache.")

    # Rasterio stuff ###########################################################

    def _get_map_dataset(self, location: str):  # type: ignore
        """
        Returns a *context manager* for the map dataset (includes all the layers).
        Extract the result in a "with ... as ...:" line.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :return: A *context manager* for the map dataset (includes all the layers).
        """
        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)

        # Save the gpkg file to disk.
        self._blob_store.save_to_disk(rel_path)

        return rasterio.open(path_on_disk)

    def get_layer_dataset(self, location: str, layer_name: str):  # type: ignore
        """
        Returns a *context manager* for the layer dataset.
        Extract the result in a "with ... as ...:" line.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        :return: A *context manager* for the layer dataset.
        """
        with self._get_map_dataset(location) as map_dataset:
            layer_dataset_path = next((path for path in map_dataset.subdatasets
                                       if path.endswith(":" + layer_name)),
                                      None)
            if layer_dataset_path is None:
                raise ValueError(f"Layer '{layer_name}' not found in map '{location}', "
                                 f"version '{self.get_version(location)}'")

            return rasterio.open(layer_dataset_path)

    def get_raster_layer_names(self, location: str) -> Sequence[str]:
        """
        Gets the list of available layers for a given map location.
        :param location: The layers name for this map location will be returned.
        :return: List of available raster layers.
        """
        all_layers_dataset = self._get_map_dataset(location)
        fully_qualified_layer_names = all_layers_dataset.subdatasets
        # fully_qualified_layer_names is a list of strings like:
        # ["GPKG:/file/path/map.gpkg:drivable_area", "GPKG:/file/path/map.gpkg:intersection", ...]
        # The layer name is everything after the last colon.

        return [name.split(":")[-1] for name in fully_qualified_layer_names]

    # Saving and loading. ###################################

    def get_gpkg_path_and_store_on_disk(self, location: str) -> str:
        """
        Saves a gpkg map from a location to disk.
        :param location: The layers name for this map location will be returned.
        :return: Path on disk to save a gpkg file.
        """
        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        self._blob_store.save_to_disk(rel_path)

        return path_on_disk

    def get_metadata_json_path_and_store_on_disk(self, location: str) -> str:
        """
        Saves a metadata.json for a location to disk.
        :param location: The layers name for this map location will be returned.
        :return: Path on disk to save metadata.json.
        """
        rel_path = self._get_metadata_json_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        self._blob_store.save_to_disk(rel_path)

        return path_on_disk

    def _get_gpkg_file_path(self, location: str) -> str:
        """
        Gets path to the gpkg map file.
        :param location: Location for which gpkg needs to be loaded.
        :return: Path to the gpkg file.
        """
        version = self.get_version(location)

        return f"{location}/{version}/map.gpkg"

    def _get_metadata_json_path(self, location: str) -> str:
        """
        Gets path to the metadata json file.
        :param location: Location for which json needs to be loaded.
        :return: Path to the meta json file.
        """
        version = self.get_version(location)

        return f"{location}/{version}/metadata.json"

    def _get_layer_matrix_npy_path(self, location: str, layer_name: str) -> str:
        """
        Gets path to the numpy file for the layer.
        :param location: Location for which layer needs to be loaded.
        :param layer_name: Which layer to load.
        :return: Path to the numpy file.
        """
        version = self.get_version(location)

        return f"{location}/{version}/{layer_name}.npy"

    def _get_distance_matrix_npy_path(self, location: str, layer_name: str) -> str:
        """
        Gets path to the distance file for the layer.
        :param location: Location for which layer needs to be loaded.
        :param layer_name: Which layer to load.
        :return: Path to the numpy file.
        """
        version = self.get_version(location)

        return f"{location}/{version}/{layer_name}_dist.npy"

    @staticmethod
    def _wait_for_expected_filesize(path_on_disk: str, expected_size: int) -> None:
        """
        Waits until the file at `path_on_disk` is exactly `expected_size` bytes.
        We wait 15 minutes before throwing an exception in case the download is just slow.
        :param path_on_disk: Path on disk to a file.
        :param expected_size: Expected size in bytes.
        """
        max_attempts = 360
        seconds_between_attempts = 5
        attempts_so_far = 0
        # Wait if file not downloaded.
        while os.path.getsize(path_on_disk) != expected_size and attempts_so_far < max_attempts:
            time.sleep(seconds_between_attempts)
            attempts_so_far += 1

        if os.path.getsize(path_on_disk) != expected_size:
            raise GPKGMapsDBException(f"Waited {max_attempts * seconds_between_attempts} seconds for "
                                      f"file {path_on_disk} to reach {expected_size}, "
                                      f"but size is now {os.path.getsize(path_on_disk)}")

    @staticmethod
    def _get_np_array(path_on_disk: str, shape: List[int], dtype: np.dtype) -> np.ndarray:  # type: ignore
        """
        Gets numpy array from file.
        :param path_on_disk: Path to numpy file.
        :param shape: Shape of layer.
        :param dtype: Numpy Dtype to use for loading numpy file.
        :return: Numpy array containing the layer.
        """
        with open(path_on_disk, "rb") as fp:
            memory_map = mmap(fp.fileno(), 0, prot=PROT_READ)
        np_data = np.ndarray(shape, dtype=dtype, buffer=memory_map)  # type: ignore

        return np_data

    def _get_expected_file_size(self, path: str, shape: List[int]) -> int:
        """
        Gets the expected file size.
        :param path: Path to the file.
        :param shape: The shape of the map file.
        :return: The expected file size.
        """
        if path.endswith('_dist.npy'):
            return shape[0] * shape[1] * 4  # float32 values take 4 bytes per pixel.
        return shape[0] * shape[1]

    def _safe_save_layer(self, location: str, layer_name: str, npy_file_path: str) -> None:
        """
        The safe way to save a map layer.
        :param location: Map location name.
        :param layer_name: Map layer name.
        :param npy_file_path: Path to the numpy file.
        """

        # Create a lock file
        layer_lock_file = os.path.join(self._map_root, f"{location}_{layer_name}.lock")
        fd = open(layer_lock_file, 'w')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            _ = self._blob_store.save_to_disk(npy_file_path, check_for_compressed=True)
        finally:
            # Delete and release the lock
            os.unlink(layer_lock_file)
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    def _get_layer_matrix(self, location: str, layer_name: str) -> npt.NDArray[np.uint8]:
        """
        Returns the map layer for `location` and `layer_name` as a numpy array.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        :return: Numpy representation of layer.
        """

        rel_path = self._get_layer_matrix_npy_path(location, layer_name)
        path_on_disk = os.path.join(self._map_root, rel_path)
        shape = self._get_shape(location, layer_name)

        if os.getenv('NUPLAN_DATA_STORE', 'local') != "local" and not os.path.exists(path_on_disk):
            self._safe_save_layer(location=location, layer_name=layer_name, npy_file_path=rel_path)
            expected_file_size = self._get_expected_file_size(rel_path, shape)
            self._wait_for_expected_filesize(path_on_disk, expected_file_size)

        return self._get_np_array(path_on_disk, shape, dtype=np.uint8)  # type: ignore

    def _get_distance_matrix(self, location: str, layer_name: str) -> npt.NDArray[np.float32]:
        """
        Returns the distance matrix for `location` and `layer_name` as a numpy array.
        Downloads the data if not already present on disk.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        :return: Numpy representation of distance matrix.
        """
        rel_path = self._get_distance_matrix_npy_path(location, layer_name)
        path_on_disk = os.path.join(self._map_root, rel_path)
        shape = self._get_shape(location, layer_name)

        if os.getenv('NUPLAN_DATA_STORE', 'local') != "local" and not os.path.exists(path_on_disk):
            self._safe_save_layer(layer_name=layer_name, location=location, npy_file_path=rel_path)
            expected_file_size = self._get_expected_file_size(rel_path, shape)
            self._wait_for_expected_filesize(path_on_disk, expected_file_size)

        return self._get_np_array(path_on_disk, shape, dtype=np.float32)  # type: ignore

    @staticmethod
    def _save_to_memmap_file(temp_file_name: str, np_arr: np.ndarray) -> None:  # type: ignore
        """
        Creates a memory-map to an array and stores in a binary file on disk.
        :param temp_file_name: The temporary file name.
        :param np_arr: The numpy array to be stored.
        """

        fp = np.memmap(temp_file_name, dtype=np_arr.dtype, mode='w+', shape=np_arr.shape)  # type: ignore
        fp[:] = np_arr[:]
        fp.flush()

    @staticmethod
    def _create_gzip_file(temp_file_name: str, gzip_file_name: str) -> None:
        """
        Stores the data in a gzip-compressed binary file.
        :param temp_file_name: The original temporary file name.
        :param gzip_file_name: The file name for gzip file.
        """

        with open(temp_file_name, 'rb') as orig_file:
            with gzip.open(gzip_file_name, 'wb') as zipped_file:
                zipped_file.writelines(orig_file)

    def _save_layer_matrix(self, location: str, layer_name: str) -> None:
        """
        Extracts the data for `layer_name` from the GPKG file for `location`,
        and saves it on disk so it can be retrieved with `_get_layer_matrix`.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        is_bin = self.is_binary(layer_name)
        with self.get_layer_dataset(location, layer_name) as layer_dataset:
            layer_data = layer_dataset_ops.load_layer_as_numpy(layer_dataset, is_bin)

        # Convert distance_px to dist matrix.
        if '_distance_px' in layer_name:
            transform_matrix = self._get_transform_matrix(location, layer_name)
            precision = 1 / transform_matrix[0, 0]

            layer_data = np.negative(layer_data / precision).astype('float32')
            layer_name = layer_name.replace('_distance_px', '_dist')

        with TemporaryDirectory() as tmp_dir:
            temp_memmap_file = os.path.join(tmp_dir, f"{layer_name}.npy")
            self._save_to_memmap_file(temp_memmap_file, layer_data)

            gzip_file = os.path.join(self._map_root,
                                     f"{location}/{self.get_version(location)}/{layer_name}.npy.gzip")
            self._create_gzip_file(temp_memmap_file, gzip_file)

    def _save_all_layers(self, location: str) -> None:
        """
        Saves data on disk for all layers in the GPKG file for `location`.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        """
        rasterio_layers = self.get_raster_layer_names(location)
        for layer_name in rasterio_layers:
            logger.debug("Working on layer: ", layer_name)
            self._save_layer_matrix(location, layer_name)
