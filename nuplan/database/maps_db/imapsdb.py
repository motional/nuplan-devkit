from abc import ABC, abstractmethod
from typing import Sequence

import geopandas as gpd

from nuplan.database.maps_db.layer import MapLayer


class IMapsDB(ABC):
    """
    Interface for MapsDB implementations.
    """

    @abstractmethod
    def load_layer(self, location: str, layer_name: str) -> MapLayer:
        """
        This is the main API method of GPKGMapsDB. It returns a MapLayer instance for the desired location and
        layer. Use `self.get_locations()` for a list of locations. The MapLayer can then be used to access crops
        and do filtering as needed on the map layer. Note also that this method will fetch resources as needed from
        remote store.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        :return: Maplayer instance.
        """
        pass

    @abstractmethod
    def layer_names(self, location: str) -> Sequence[str]:
        """
        Gets the list of available layers for a given map location.
        :param location: The layers name for this map location will be returned.
        :return: List of available layers.
        """
        pass

    @abstractmethod
    def load_vector_layer(self, location: str, layer_name: str) -> gpd.geodataframe:
        """
        Loads Vector Layer.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :param layer_name: Name of layer, e.g. `drivable_area`. Use self.layer_names(location) for complete list.
        """
        pass

    @abstractmethod
    def vector_layer_names(self, location: str) -> Sequence[str]:
        """
        Gets list of all available vector layers.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :return: List of available vector layers.
        """
        pass

    @abstractmethod
    def purge_cache(self) -> None:
        """
        Purges cache in the data root.
        """
        pass

    @abstractmethod
    def get_version(self, location: str) -> str:
        """
        Gets version of location.
        :param location: Name of map location, e.g. "sg-one-north`. See `self.get_locations()`.
        :return: The location version for the map location.
        """
        pass

    @abstractmethod
    def get_map_version(self) -> str:
        """
        Gets version of a set of maps.
        :return: The map version of this iMapsDB.
        """
        pass
