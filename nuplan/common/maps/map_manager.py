from typing import Dict

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_factory import AbstractMapFactory


class MapManager:
    """Class to store created maps using a map factory."""

    def __init__(self, map_factory: AbstractMapFactory):
        """
        Constructor of MapManager.
        :param map_factory: map factory.
        """
        self.map_factory = map_factory
        self.maps: Dict[str, AbstractMap] = {}

    def get_map(self, map_name: str) -> AbstractMap:
        """
        Returns the queried map from the map factory, creating it if it's missing.
        :param map_name: Name of the map.
        :return: The queried map.
        """
        if map_name not in self.maps:
            self.maps[map_name] = self.map_factory.build_map_from_name(map_name)

        return self.maps[map_name]
