from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Set, Tuple

import numpy as np


@dataclasses.dataclass
class MapLayerMeta:
    """Stores the metadata for a map layer (layer name and md5 hash)."""

    def __init__(self, name: str, md5_hash: str, can_dilate: bool, is_binary: bool, precision: float):
        """
        Initializes MapLayerMeta.
        :param name: Map layer name, e.g. 'drivable_area'
        :param md5_hash: Hash calculated from the mask itself.
        :param can_dilate: Whether we support dilation for this layer.
        :param is_binary: Whether the layer is binary. Most layers, e.g. `drivable_area` are. But some,
            like `intensity` are not.
        :param precision: Identified map resolution in meters per pixel. Typically set to 0.1, meaning that 10 pixels
            correspond to 1 meter.
        """
        self.name = name
        self.md5_hash = md5_hash
        self.can_dilate = can_dilate
        self.is_binary = is_binary
        self.precision = precision

    @property
    def binary_mask_name(self) -> str:
        """
        Returns the binary mask file name.
        :return: The binary mask file name.
        """
        return self.md5_hash + '.bin'

    @property
    def binary_joint_dist_name(self) -> str:
        """
        Returns the binary joint distance file name.
        :return: The binary joint distance file name.
        """
        return self.md5_hash + '.joint_dist.bin'

    @property
    def png_mask_name(self) -> str:
        """
        Returns the PNG mask file name.
        :return: The PNG mask file name.
        """
        return self.md5_hash + '.png'

    def serialize(self) -> Dict[str, Any]:
        """
        Serializes the meta data of a map layer to a JSON-friendly dictionary representation.
        :return: A dict of meta data of map layer.
        """
        return {
            'name': self.name,
            'md5_hash': self.md5_hash,
            'can_dilate': self.can_dilate,
            'is_binary': self.is_binary,
            'precision': self.precision,
        }

    @classmethod
    def deserialize(cls, encoding: Dict[str, Any]) -> MapLayerMeta:
        """
        Instantiates a MapLayerMeta instance from serialized dictionary representation.
        :param encoding: Output from serialize.
        :return: Deserialized meta data.
        """
        return MapLayerMeta(
            name=encoding['name'],
            md5_hash=encoding['md5_hash'],
            can_dilate=encoding['can_dilate'],
            is_binary=encoding['is_binary'],
            precision=encoding['precision'],
        )


@dataclasses.dataclass
class MapVersionMeta:
    """Stores the metadata for a MapVersionMeta, a collection of MapLayerMeta objects."""

    def __init__(self, name: str) -> None:
        """
        Constructor.
        :param name: The name of a map layer.
        """
        self.name = name
        self.size = None  # Tuple[int, int]
        self.layers: Dict[str, MapLayerMeta] = {}  # Dict[layer_name: layer]
        self.origin = None  # Tuple[float, float]
        self.transform_matrix = None  # Optional[np.ndarray (4x4)]

    def __getitem__(self, item: str) -> MapLayerMeta:
        """
        Retrieves the MapLayer meta data for a given layer name.
        :param item: Layer name.
        :return: The metadata of a map layer.
        """
        return self.layers[item]

    def set_size(self, size: Tuple[int, int]) -> None:
        """
        Sets the size of map layer.
        :param size: The size used to set the map layer.
        """
        if self.size is None:
            self.size = size  # type: ignore
        else:
            assert size == self.size, "Map layer size doesn't match map other layers from this map version."

    def set_map_origin(self, origin: Tuple[float, float]) -> None:
        """
        Sets the origin of the map.
        :param origin: The coordinate of the map origin.
        """
        if self.origin is None:
            self.origin = origin  # type: ignore
        else:
            assert origin == self.origin, f"origin does not match other layers for map version {self.name}"

    def set_transform_matrix(self, transform_matrix: List[List[float]]) -> None:
        """
        Sets the transform matrix of the MapVersionMeta object.
        :param transform_matrix: The transform matrix for converting from physical coordinates to pixel coordinates.
        """
        if transform_matrix is not None:
            self.transform_matrix = np.array(transform_matrix)  # type: ignore

    def add_layer(self, map_layer: MapLayerMeta) -> None:
        """
        Adds layer to the MapLayerMeta.
        :param map_layer: The map layer to be added.
        """
        self.layers[map_layer.name] = map_layer

    @property
    def layer_names(self) -> List[str]:
        """
        Returns a list of the layer names.
        :return: A list of the layer names.
        """
        return sorted(list(self.layers.keys()))

    def serialize(self) -> Dict[str, Any]:
        """
        Serializes the MapVersionMeta instance to a JSON-friendly dictionary representation.
        :return: Encoding of the MapVersionMeta.
        """
        return {
            'size': self.size,
            'name': self.name,
            'origin': self.origin,
            'layers': [layer.serialize() for layer in self.layers.values()],
        }

    @classmethod
    def deserialize(cls, encoding: Dict[str, Any]) -> MapVersionMeta:
        """
        Instantiates a MapVersionMeta instance from serialized dictionary representation.
        :param encoding: Output from serialize.
        :return: Deserialized MapVersionMeta.
        """
        mv = MapVersionMeta(name=encoding['name'])
        mv.set_size(encoding['size'])

        # Not every map version has an origin or transformation matrix.
        mv.set_map_origin(encoding.get('origin'))  # type: ignore
        mv.set_transform_matrix(encoding.get('transform_matrix'))  # type: ignore

        for layer in encoding['layers']:
            mv.add_layer(MapLayerMeta.deserialize(layer))

        return mv

    def __hash__(self) -> int:
        """
        Returns the hash value for the MapVersionMeta object.
        :return: The hash value.
        """
        return hash((self.name, *[(key, self.layers[key].md5_hash) for key in sorted(self.layers)]))

    def __eq__(self, other: object) -> bool:
        """
        Compares two MapVersionMeta objects are the same or not by checking the hash value.
        :param other: The other MapVersionMeta objects.
        :return: True if both objects are the same, otherwise False.
        """
        if not isinstance(other, MapVersionMeta):
            return NotImplemented

        return self.__hash__() == hash(other)


@dataclasses.dataclass
class MapMetaData:
    """Stores the map metadata for all the MapVersions."""

    def __init__(self) -> None:
        """Init function for class."""
        self.versions: Dict[str, MapVersionMeta] = {}  # Maps the map version name to MapVersionMeta object.

    def __getitem__(self, item: str) -> MapVersionMeta:
        """
        Retrieves the MapVersionMeta for a given map version name.
        :param item: Map version name.
        :return: A MapVersionMeta object.
        """
        return self.versions[item]

    def add_version(self, map_version: MapVersionMeta) -> None:
        """
        Adds a MapVersionMeta to the versions.
        :param map_version: A map version to be added.
        """
        self.versions[map_version.name] = map_version

    @property
    def hash_sizes(self) -> Set[Tuple[str, Tuple[int, int]]]:
        """Returns the hash size of each layer in each map version."""
        hash_sizes_: Set[Tuple[str, Tuple[int, int]]] = set()
        for version in self.versions.values():
            for layer in version.layers.values():
                hash_sizes_.add((layer.md5_hash, tuple(version.size)))  # type: ignore

        return hash_sizes_

    @property
    def version_names(self) -> List[str]:
        """
        Returns a list of version names.
        :return: A list of version names.
        """
        return sorted(list(self.versions.keys()))

    def serialize(self) -> List[Dict[str, Any]]:
        """
        Serializes the MapMetaData instance to a JSON-friendly list representation.
        :return: Encoding of the MapMetaData.
        """
        return [map_version.serialize() for map_version in self.versions.values()]

    @classmethod
    def deserialize(cls, encoding: List[Dict[str, Any]]) -> MapMetaData:
        """
        Instantiates a MapMetaData instance from serialized list representation.
        :param encoding: Output from serialize.
        :return: Deserialized MapMetaData.
        """
        mmd = MapMetaData()
        for map_version_encoding in encoding:
            mmd.add_version(MapVersionMeta.deserialize(map_version_encoding))
        return mmd
