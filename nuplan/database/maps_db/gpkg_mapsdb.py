import fcntl
import glob
import json
import logging
import os
import time
import warnings
from functools import lru_cache
from typing import Any, List, Sequence, Tuple, Type

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pyogrio
import rasterio

from nuplan.common.utils.s3_utils import get_s3_client
from nuplan.database.common.blob_store.creator import BlobStoreCreator
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.maps_db import layer_dataset_ops
from nuplan.database.maps_db.imapsdb import IMapsDB
from nuplan.database.maps_db.layer import MapLayer
from nuplan.database.maps_db.metadata import MapLayerMeta

logger = logging.getLogger(__name__)

# To silence NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Available map locations
MAP_LOCATIONS = {'sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood'}

# Dimensions of raster layers for each location
MAP_DIMENSIONS = {
    'sg-one-north': (21070, 28060),
    'us-ma-boston': (20380, 28730),
    'us-nv-las-vegas-strip': (69820, 30120),
    'us-pa-pittsburgh-hazelwood': (22760, 23090),
}

# S3 download params.
MAX_ATTEMPTS = 360
SECONDS_BETWEEN_ATTEMPTS = 5

# Dummy layer to use for downloading the map package for the first time
DUMMY_LOAD_LAYER = 'lane_connectors'


class GPKGMapsDBException(Exception):
    """GPKGMapsDB Exception Class."""

    def __init__(self, message: str) -> None:
        """
        Constructor.
        :param message: Exception message.
        """
        super().__init__(message)


class GPKGMapsDB(IMapsDB):
    """GPKG MapsDB implementation."""

    def __init__(self, map_version: str, map_root: str) -> None:
        """
        Constructor.
        :param map_version: Version of map.
        :param map_root: Root folder of the maps.
        """
        self._map_version = map_version
        self._map_root = map_root

        self._blob_store = BlobStoreCreator.create_mapsdb(map_root=self._map_root)
        version_file = self._blob_store.get(f"{self._map_version}.json")  # get blob and save to disk
        self._metadata = json.load(version_file)

        # The dimensions of the maps are hard-coded for the 4 locations.
        self._map_dimensions = MAP_DIMENSIONS

        # S3 file download parameters.
        self._max_attempts = MAX_ATTEMPTS
        self._seconds_between_attempts = SECONDS_BETWEEN_ATTEMPTS

        self._map_lock_dir = os.path.join(self._map_root, '.maplocks')
        os.makedirs(self._map_lock_dir, exist_ok=True)

        # Load map data to trigger automatic downloading.
        self._load_map_data()

    def __reduce__(self) -> Tuple[Type['GPKGMapsDB'], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        This object is reconstructed by pickle to avoid serializing potentially large state/caches.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._map_version, self._map_root)

    def _load_map_data(self) -> None:
        """Load all available maps once to trigger automatic downloading if the maps are loaded for the first time."""
        # TODO: Spawn multiple threads for parallel downloading
        for location in MAP_LOCATIONS:
            self.load_vector_layer(location, DUMMY_LOAD_LAYER)

    @property
    def version_names(self) -> List[str]:
        """
        Lists the map version names for all valid map locations, e.g.
        ['9.17.1964', '9.12.1817', '9.15.1915', '9.17.1937']
        """
        return [self._metadata[location]["version"] for location in self.get_locations()]

    def get_map_version(self) -> str:
        """Inherited, see superclass."""
        return self._map_version

    def get_version(self, location: str) -> str:
        """Inherited, see superclass."""
        return str(self._metadata[location]["version"])

    def _get_shape(self, location: str, layer_name: str) -> List[int]:
        """
        Gets the shape of a layer given the map location and layer name.
        :param location: Name of map location, e.g. "sg-one-north". See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        if layer_name == 'intensity':
            return self._metadata[location]["layers"]["Intensity"]["shape"]  # type: ignore
        else:
            # The dimensions of other map layers are using the hard-coded values.
            return list(self._map_dimensions[location])

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

    def get_locations(self) -> Sequence[str]:
        """
        Gets the list of available location in this GPKGMapsDB version.
        """
        return self._metadata.keys()  # type: ignore

    def layer_names(self, location: str) -> Sequence[str]:
        """Inherited, see superclass."""
        gpkg_layers = self._metadata[location]["layers"].keys()
        return list(filter(lambda x: '_distance_px' not in x, gpkg_layers))

    def load_layer(self, location: str, layer_name: str) -> MapLayer:
        """Inherited, see superclass."""
        if layer_name == "intensity":
            layer_name = "Intensity"

        is_bin = self.is_binary(layer_name)
        can_dilate = self._can_dilate(layer_name)
        layer_data = self._get_layer_matrix(location, layer_name)
        transform_matrix = self._get_transform_matrix(location, layer_name)

        # We assume that the map's pixel-per-meter ratio is the same in the x and y directions,
        # since the MapLayer class requires that.
        precision = 1 / transform_matrix[0, 0]
        layer_meta = MapLayerMeta(
            name=layer_name,
            md5_hash="not_used_for_gpkg_mapsdb",
            can_dilate=can_dilate,
            is_binary=is_bin,
            precision=precision,
        )

        distance_matrix = None

        return MapLayer(
            data=layer_data, metadata=layer_meta, joint_distance=distance_matrix, transform_matrix=transform_matrix
        )

    def _wait_for_expected_filesize(self, path_on_disk: str, location: str) -> None:
        """
        Waits until the file at `path_on_disk` is exactly `expected_size` bytes.
        :param path_on_disk: Path of the file being downloaded.
        :param location: Location to which the file belongs.
        """
        if isinstance(self._blob_store, LocalStore):
            return

        s3_bucket = self._blob_store._remote._bucket
        s3_key = os.path.join(self._blob_store._remote._prefix, self._get_gpkg_file_path(location))

        # Create a new S3 session for the request.
        # In a multiprocess context, using the pickled / unpickled client can lead to authentication failures.
        client = get_s3_client()
        map_file_size = client.head_object(Bucket=s3_bucket, Key=s3_key).get('ContentLength', 0)

        # Wait if file not downloaded.
        for _ in range(self._max_attempts):
            if os.path.getsize(path_on_disk) == map_file_size:
                break

            time.sleep(self._seconds_between_attempts)

        if os.path.getsize(path_on_disk) != map_file_size:
            raise GPKGMapsDBException(
                f"Waited {self._max_attempts * self._seconds_between_attempts} seconds for "
                f"file {path_on_disk} to reach {map_file_size}, "
                f"but size is now {os.path.getsize(path_on_disk)}"
            )

    def _safe_save_layer(self, layer_lock_file: str, file_path: str) -> None:
        """
        Safely download the file.
        :param layer_lock_file: Path to lock file.
        :param file_path: Path of the file being downloaded.
        """
        fd = open(layer_lock_file, 'w')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            _ = self._blob_store.save_to_disk(file_path, check_for_compressed=True)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    # The size of the cache was derived from testing with the Raster Model
    #   on our cluster to balance memory usage and performance.
    @lru_cache(maxsize=16)
    def load_vector_layer(self, location: str, layer_name: str) -> gpd.geodataframe:
        """Inherited, see superclass."""
        # TODO: Remove temporary workaround once map_version is cleaned
        location = location.replace('.gpkg', '')

        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)

        if not os.path.exists(path_on_disk):
            layer_lock_file = f'{self._map_lock_dir}/{location}_{layer_name}.lock'
            self._safe_save_layer(layer_lock_file, rel_path)
        self._wait_for_expected_filesize(path_on_disk, location)

        with warnings.catch_warnings():
            # Suppress the warnings from the GPKG operations below so that they don't spam the training logs.
            warnings.filterwarnings("ignore")

            # The projected coordinate system depends on which UTM zone the mapped location is in.
            map_meta = gpd.read_file(path_on_disk, layer="meta", engine="pyogrio")
            projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]

            gdf_in_pixel_coords = pyogrio.read_dataframe(path_on_disk, layer=layer_name, fid_as_index=True)
            gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)

            # For backwards compatibility, cast the index to string datatype.
            #   and mirror it to the "fid" column.
            gdf_in_utm_coords.index = gdf_in_utm_coords.index.map(str)
            gdf_in_utm_coords["fid"] = gdf_in_utm_coords.index

        return gdf_in_utm_coords

    def vector_layer_names(self, location: str) -> Sequence[str]:
        """Inherited, see superclass."""
        # TODO: Remove temporary workaround once map_version is cleaned
        location = location.replace('.gpkg', '')

        rel_path = self._get_gpkg_file_path(location)
        path_on_disk = os.path.join(self._map_root, rel_path)
        self._blob_store.save_to_disk(rel_path)

        return pyogrio.list_layers(path_on_disk)  # type: ignore

    def purge_cache(self) -> None:
        """Inherited, see superclass."""
        logger.debug("Purging cache...")
        for f in glob.glob(os.path.join(self._map_root, "gpkg", "*")):
            os.remove(f)
        logger.debug("Done purging cache.")

    def _get_map_dataset(self, location: str) -> rasterio.DatasetReader:
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

    def get_layer_dataset(self, location: str, layer_name: str) -> rasterio.DatasetReader:
        """
        Returns a *context manager* for the layer dataset.
        Extract the result in a "with ... as ...:" line.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        :return: A *context manager* for the layer dataset.
        """
        with self._get_map_dataset(location) as map_dataset:
            layer_dataset_path = next(
                (path for path in map_dataset.subdatasets if path.endswith(":" + layer_name)), None
            )
            if layer_dataset_path is None:
                raise ValueError(
                    f"Layer '{layer_name}' not found in map '{location}', " f"version '{self.get_version(location)}'"
                )

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

        return f"{location}/{version}/{layer_name}.npy.npz"

    @staticmethod
    def _get_np_array(path_on_disk: str) -> np.ndarray:  # type: ignore
        """
        Gets numpy array from file.
        :param path_on_disk: Path to numpy file.
        :return: Numpy array containing the layer.
        """
        np_data = np.load(path_on_disk)

        return np_data['data']  # type: ignore

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

    def _get_layer_matrix(self, location: str, layer_name: str) -> npt.NDArray[np.uint8]:
        """
        Returns the map layer for `location` and `layer_name` as a numpy array.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        :return: Numpy representation of layer.
        """
        rel_path = self._get_layer_matrix_npy_path(location, layer_name)
        path_on_disk = os.path.join(self._map_root, rel_path)

        if not os.path.exists(path_on_disk):
            self._save_layer_matrix(location=location, layer_name=layer_name)

        return self._get_np_array(path_on_disk)

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

        npy_file_path = os.path.join(self._map_root, f"{location}/{self.get_version(location)}/{layer_name}.npy")
        np.savez_compressed(npy_file_path, data=layer_data)

    def _save_all_layers(self, location: str) -> None:
        """
        Saves data on disk for all layers in the GPKG file for `location`.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        """
        rasterio_layers = self.get_raster_layer_names(location)
        for layer_name in rasterio_layers:
            logger.debug("Working on layer: ", layer_name)
            self._save_layer_matrix(location, layer_name)
