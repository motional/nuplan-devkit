from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, Type

from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.maps_db.imapsdb import IMapsDB


class NuPlanMapFactory(AbstractMapFactory):
    """
    Factory creating maps from an IMapsDB interface.
    """

    def __init__(self, maps_db: IMapsDB):
        """
        :param maps_db: An IMapsDB instance e.g. GPKGMapsDB.
        """
        self._maps_db = maps_db

    def __reduce__(self) -> Tuple[Type[NuPlanMapFactory], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._maps_db,)

    def build_map_from_name(self, map_name: str) -> NuPlanMap:
        """
        Builds a map interface given a map name.
        Examples of names: 'sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood'
        :param map_name: Name of the map.
        :return: The constructed map interface.
        """
        return NuPlanMap(self._maps_db, map_name.replace(".gpkg", ""))


@lru_cache(maxsize=2)
def get_maps_db(map_root: str, map_version: str) -> GPKGMapsDB:
    """
    Get a maps_db from disk.
    :param map_root: The root folder for the map data.
    :param map_version: The version of the map to load.
    :return; The loaded MapsDB object.
    """
    return GPKGMapsDB(map_root=map_root, map_version=map_version)


@lru_cache(maxsize=32)
def get_maps_api(map_root: str, map_version: str, map_name: str) -> NuPlanMap:
    """
    Get a NuPlanMap object corresponding to a particular set of parameters.
    :param map_root: The root folder for the map data.
    :param map_version: The map version to load.
    :param map_name: The map name to load.
    :return: The loaded NuPlanMap object.
    """
    maps_db = get_maps_db(map_root, map_version)
    return NuPlanMap(maps_db, map_name.replace(".gpkg", ""))
