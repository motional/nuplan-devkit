import abc

from nuplan.common.maps.abstract_map import AbstractMap


class AbstractMapFactory(abc.ABC):
    """
    Generic map factory interface.
    """

    def build_map_from_name(self, map_name: str) -> AbstractMap:
        """
        Builds a map interface given a map name.
        :param map_name: Name of the map.
        :return: The constructed map interface.
        """
        pass
