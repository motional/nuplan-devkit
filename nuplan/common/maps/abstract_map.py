from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map_objects import Crosswalk, Intersection, Lane, LaneConnector, StopLine
from nuplan.common.maps.maps_datatypes import LaneSegmentConnections, LaneSegmentCoords, LaneSegmentMetaData, \
    RasterLayer, RasterMap, SemanticMapLayer

MapObject = Union[Lane, LaneConnector, Crosswalk, Intersection, StopLine]


class AbstractMap(abc.ABC):
    """
    Interface for generic scenarios Map API.
    """
    @abc.abstractmethod
    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """
        Returns the available map objects types
        :return: A list of SemanticMapLayers
        """
        pass

    @abc.abstractmethod
    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        """
        Returns the available map objects types
        :return: A list of SemanticMapLayers
        """
        pass

    @abc.abstractmethod
    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        """
        Gets multiple raster maps specified
        :param layer: A semantic layer to query.
        :return: RasterMap. A dictionary mapping SemanticMapLayer to a RasterLayer
        """
        pass

    @abc.abstractmethod
    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        """
        Gets multiple raster maps specified
        :param layers: A list of semantic layers.
        :return: RasterMap. A dictionary mapping SemanticMapLayer to a RasterLayer
        """
        pass

    @property
    @abc.abstractmethod
    def map_name(self) -> str:
        """
        :return: name of the location where the map is.
        """
        pass

    @abc.abstractmethod
    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        """
        Returns all map objects on a semantic layer that contains the given point x, y
        :param point: [m] x, y coordinates in global frame
        :param layer: A semantic layer to query.
        :return: list of map objects
        """
        pass

    @abc.abstractmethod
    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
        """
        Returns one map objects on a semantic layer that contains the given point x, y
        :param point: [m] x, y coordinates in global frame
        :param layer: A semantic layer to query.
        :return: list of map objects
        @raise AssertionError if more than one object is found
        """
        pass

    @abc.abstractmethod
    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        """
        Checks if the given point x, y lies within a semantic layer
        :param point: [m] x, y coordinates in global frame
        :param layer: A semantic layer to query.
        :return: True if [x, y] is in a layer, False if it is not
        """
        pass

    @abc.abstractmethod
    def get_proximal_map_objects(self, point: Point2D, radius: float, layers: List[SemanticMapLayer]) \
            -> Dict[SemanticMapLayer, List[MapObject]]:
        """
        Extract map objects within the given radius around the point x, y
        :param point: [m] x, y coordinates in global frame
        :param radius [m] floating number about vector map query range
        :param layers: desired layers to check
        :return: A dictionary mapping SemanticMapLayers to lists of map objects
        """
        pass

    @abc.abstractmethod
    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> MapObject:
        """
        Gets the lane with the given lane id
        :param object_id: desired unique id of a lane that should be extracted
        :param layer: A semantic layer to query.
        :return: a map object
        """
        pass

    @abc.abstractmethod
    def get_distance_to_nearest_map_object(self, point: Point2D, layer: SemanticMapLayer) \
            -> Optional[Tuple[str, float]]:
        """
        Gets the distance (in meters) to the nearest desired surface; that distance is the L1 norm from the point to
        the closest location on the surface.
        :param point: [m] x, y coordinates in global frame
        :param layer: A semantic layer to query..
        :return: The surface ID and the distance to the surface if there is one. If there isn't, then -1 and np.NaN will
            be returned for the surface ID and distance to the surface respectively.
        """
        pass

    @abc.abstractmethod
    def get_neighbor_vector_map(self, point: Point2D, radius: float) -> \
            Tuple[LaneSegmentCoords, LaneSegmentConnections, LaneSegmentMetaData]:
        """
        Extract neighbor vector maps information around ego vehicle
        :param point: [m] x, y coordinates in global frame
        :param radius [m] floating number about vector map query range
        :return
            ls_coords: lane_segment coords in shape of [num_lane_segment, 2, 2]
            ls_conns: lane_segment connections [start_idx, end_idx] in shape of [num_connection, 2]
            ls_meta: lane_segment meta info including in_intersection, turn_direction,etc
        """
        pass
