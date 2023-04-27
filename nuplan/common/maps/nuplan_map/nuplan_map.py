from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

import geopandas
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as geom

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map_objects import (
    Intersection,
    Lane,
    LaneConnector,
    RoadBlockGraphEdgeMapObject,
    StopLine,
)
from nuplan.common.maps.maps_datatypes import RasterLayer, RasterMap, SemanticMapLayer, StopLineType, VectorLayer
from nuplan.common.maps.nuplan_map.intersection import NuPlanIntersection
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.nuplan_map.polygon_map_object import NuPlanPolygonMapObject
from nuplan.common.maps.nuplan_map.roadblock import NuPlanRoadBlock
from nuplan.common.maps.nuplan_map.roadblock_connector import NuPlanRoadBlockConnector
from nuplan.common.maps.nuplan_map.stop_line import NuPlanStopLine
from nuplan.common.maps.nuplan_map.utils import is_in_type, raster_layer_from_map_layer
from nuplan.database.maps_db.imapsdb import IMapsDB
from nuplan.database.maps_db.layer import MapLayer


class NuPlanMap(AbstractMap):
    """
    NuPlanMap implementation of Map API.
    """

    def __init__(self, maps_db: IMapsDB, map_name: str) -> None:
        """
        Initializes the map class.
        :param maps_db: MapsDB instance.
        :param map_name: Name of the map.
        """
        self._maps_db = maps_db
        self._vector_map: Dict[str, VectorLayer] = defaultdict(VectorLayer)
        self._raster_map: Dict[str, RasterLayer] = defaultdict(RasterLayer)
        self._map_objects: Dict[SemanticMapLayer, Dict[str, MapObject]] = defaultdict(dict)
        self._map_name = map_name

        self._map_object_getter: Dict[SemanticMapLayer, Callable[[str], MapObject]] = {
            SemanticMapLayer.LANE: self._get_lane,
            SemanticMapLayer.LANE_CONNECTOR: self._get_lane_connector,
            SemanticMapLayer.ROADBLOCK: self._get_roadblock,
            SemanticMapLayer.ROADBLOCK_CONNECTOR: self._get_roadblock_connector,
            SemanticMapLayer.STOP_LINE: self._get_stop_line,
            SemanticMapLayer.CROSSWALK: self._get_crosswalk,
            SemanticMapLayer.INTERSECTION: self._get_intersection,
            SemanticMapLayer.WALKWAYS: self._get_walkway,
            SemanticMapLayer.CARPARK_AREA: self._get_carpark_area,
        }

        self._vector_layer_mapping = {
            SemanticMapLayer.LANE: 'lanes_polygons',
            SemanticMapLayer.ROADBLOCK: 'lane_groups_polygons',
            SemanticMapLayer.INTERSECTION: 'intersections',
            SemanticMapLayer.STOP_LINE: 'stop_polygons',
            SemanticMapLayer.CROSSWALK: 'crosswalks',
            SemanticMapLayer.DRIVABLE_AREA: 'drivable_area',
            SemanticMapLayer.LANE_CONNECTOR: 'lane_connectors',
            SemanticMapLayer.ROADBLOCK_CONNECTOR: 'lane_group_connectors',
            SemanticMapLayer.BASELINE_PATHS: 'baseline_paths',
            SemanticMapLayer.BOUNDARIES: 'boundaries',
            SemanticMapLayer.WALKWAYS: 'walkways',
            SemanticMapLayer.CARPARK_AREA: 'carpark_areas',
        }
        self._raster_layer_mapping = {
            SemanticMapLayer.DRIVABLE_AREA: 'drivable_area',
        }

        # Special vector layer mapping for lane connector polygons.
        self._LANE_CONNECTOR_POLYGON_LAYER = 'gen_lane_connectors_scaled_width_polygons'

    def __reduce__(self) -> Tuple[Type['NuPlanMap'], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        This object is reconstructed by pickle to avoid serializing potentially large state/caches.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._maps_db, self._map_name)

    @property
    def map_name(self) -> str:
        """Inherited, see superclass."""
        return self._map_name

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """Inherited, see superclass."""
        return list(self._map_object_getter.keys())

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        """Inherited, see superclass."""
        return list(self._raster_layer_mapping.keys())

    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        """Inherited, see superclass."""
        layer_id = self._semantic_raster_layer_map(layer)

        return self._load_raster_layer(layer_id)

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        """Inherited, see superclass."""
        raster_map = RasterMap(layers=defaultdict(RasterLayer))

        for layer in layers:
            raster_map.layers[layer] = self.get_raster_map_layer(layer)

        return raster_map

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        """Inherited, see superclass."""
        if layer == SemanticMapLayer.TURN_STOP:
            stop_lines = self._get_vector_map_layer(SemanticMapLayer.STOP_LINE)
            in_stop_line = stop_lines.loc[stop_lines.contains(geom.Point(point.x, point.y))]

            return any(in_stop_line.loc[in_stop_line["stop_polygon_type_fid"] == StopLineType.TURN_STOP.value].values)

        return bool(is_in_type(point.x, point.y, self._get_vector_map_layer(layer)))

    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        """Inherited, see superclass."""
        try:
            return self._get_all_map_objects(point, layer)
        except KeyError:
            raise ValueError(f"Object representation for layer: {layer.name} is unavailable")

    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
        """Inherited, see superclass."""
        map_objects = self.get_all_map_objects(point, layer)

        if len(map_objects) > 1:
            raise AssertionError(
                f"{len(map_objects)} map objects found. Expected only one. " "Try using get_all_map_objects()"
            )

        if len(map_objects) == 0:
            return None

        return map_objects[0]

    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[SemanticMapLayer]
    ) -> Dict[SemanticMapLayer, List[MapObject]]:
        """Inherited, see superclass."""
        x_min, x_max = point.x - radius, point.x + radius
        y_min, y_max = point.y - radius, point.y + radius
        patch = geom.box(x_min, y_min, x_max, y_max)

        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]

        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"

        object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

        for layer in layers:
            object_map[layer] = self._get_proximity_map_object(patch, layer)

        return object_map

    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> Optional[MapObject]:
        """Inherited, see superclass."""
        try:
            if object_id not in self._map_objects[layer]:
                map_object: MapObject = self._map_object_getter[layer](object_id)
                self._map_objects[layer][object_id] = map_object

            return self._map_objects[layer][object_id]
        except KeyError:
            raise ValueError(f"Object representation for layer: {layer.name} object: {object_id} is unavailable")

    def get_distance_to_nearest_map_object(
        self, point: Point2D, layer: SemanticMapLayer
    ) -> Tuple[Optional[str], Optional[float]]:
        """Inherited from superclass."""
        surfaces = self._get_vector_map_layer(layer)

        if surfaces is not None:
            surfaces['distance_to_point'] = surfaces.apply(
                lambda row: geom.Point(point.x, point.y).distance(row.geometry), axis=1
            )
            surfaces = surfaces.sort_values(by='distance_to_point')

            # A single surface might be made up of multiple polygons (due to an old practice of annotating a long
            # surface with multiple polygons; going forward there are plans by the mapping team to update the maps such
            # that one surface is covered by at most one polygon), thus we simply pick whichever polygon is closest to
            # the point.
            nearest_surface = surfaces.iloc[0]
            nearest_surface_id = nearest_surface.fid
            nearest_surface_distance = nearest_surface.distance_to_point
        else:
            nearest_surface_id = None
            nearest_surface_distance = None

        return nearest_surface_id, nearest_surface_distance

    def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
        """Inherited from superclass"""
        raise NotImplementedError

    def get_distances_matrix_to_nearest_map_object(
        self, points: List[Point2D], layer: SemanticMapLayer
    ) -> Optional[npt.NDArray[np.float64]]:
        """
        Returns the distance matrix (in meters) between a list of points and their nearest desired surface.
            That distance is the L1 norm from the point to the closest location on the surface.
        :param points: [m] A list of x, y coordinates in global frame.
        :param layer: A semantic layer to query.
        :return: An array of shortest distance from each point to the nearest desired surface.
        """
        surfaces = self._get_vector_map_layer(layer)

        if surfaces is not None:
            # Construct geo series
            corner_points = geopandas.GeoSeries([geom.Point(point.x, point.y) for point in points])

            # Distance
            distances = surfaces.geometry.apply(lambda g: corner_points.distance(g))

            # Distance to the nearest surface
            distances = np.asarray(distances.min())
            return cast(npt.NDArray[np.float64], distances)
        else:
            return None

    def initialize_all_layers(self) -> None:
        """
        Load all layers to vector map
        :param: None
        :return: None
        """
        for layer_name in self._vector_layer_mapping.values():
            self._load_vector_map_layer(layer_name)
        for layer_name in self._raster_layer_mapping.values():
            self._load_vector_map_layer(layer_name)
        self._load_vector_map_layer(self._LANE_CONNECTOR_POLYGON_LAYER)

    def _semantic_vector_layer_map(self, layer: SemanticMapLayer) -> str:
        """
        Mapping from SemanticMapLayer int to MapsDB internal representation of vector layers.
        :param layer: The querired semantic map layer.
        :return: A internal layer name as a string.
        @raise ValueError if the requested layer does not exist for MapsDBMap
        """
        try:
            return self._vector_layer_mapping[layer]
        except KeyError:
            raise ValueError("Unknown layer: {}".format(layer.name))

    def _semantic_raster_layer_map(self, layer: SemanticMapLayer) -> str:
        """
        Mapping from SemanticMapLayer int to MapsDB internal representation of raster layers.
        :param layer: The queried semantic map layer.
        :return: A internal layer name as a string.
        @raise ValueError if the requested layer does not exist for MapsDBMap
        """
        try:
            return self._raster_layer_mapping[layer]
        except KeyError:
            raise ValueError("Unknown layer: {}".format(layer.name))

    def _get_vector_map_layer(self, layer: SemanticMapLayer) -> VectorLayer:
        """Inherited, see superclass."""
        layer_id = self._semantic_vector_layer_map(layer)
        return self._load_vector_map_layer(layer_id)

    def _load_raster_layer(self, layer_name: str) -> RasterLayer:
        """
        Load and cache raster layers.
        :layer_name: the name of the vector layer to be loaded.
        :return: the loaded RasterLayer.
        """
        if layer_name not in self._raster_map:
            map_layer: MapLayer = self._maps_db.load_layer(self._map_name, layer_name)
            self._raster_map[layer_name] = raster_layer_from_map_layer(map_layer)

        return self._raster_map[layer_name]

    def _load_vector_map_layer(self, layer_name: str) -> VectorLayer:
        """
        Load and cache vector layers.
        :layer_name: the name of the vector layer to be loaded.
        :return: the loaded VectorLayer.
        """
        if layer_name not in self._vector_map:

            if layer_name == 'drivable_area':
                self._initialize_drivable_area()
            else:
                self._vector_map[layer_name] = self._maps_db.load_vector_layer(self._map_name, layer_name)
        return self._vector_map[layer_name]

    def _get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        """
        Gets a list of lanes where its polygon overlaps the queried point.
        :param point: [m] x, y coordinates in global frame.
        :return: a list of lanes. An empty list if no lanes were found.
        """
        if layer == SemanticMapLayer.LANE_CONNECTOR:
            return self._get_all_lane_connectors(point)
        else:
            layer_df = self._get_vector_map_layer(layer)
            ids = layer_df.loc[layer_df.contains(geom.Point(point.x, point.y))]['fid'].tolist()

            return [self.get_map_object(map_object_id, layer) for map_object_id in ids]

    def _get_all_lane_connectors(self, point: Point2D) -> List[LaneConnector]:
        """
        Gets a list of lane connectors where its polygon overlaps the queried point.
        :param point: [m] x, y coordinates in global frame.
        :return: a list of lane connectors. An empty list if no lane connectors were found.
        """
        lane_connectors_df = self._load_vector_map_layer(self._LANE_CONNECTOR_POLYGON_LAYER)
        ids = lane_connectors_df.loc[lane_connectors_df.contains(geom.Point(point.x, point.y))][
            'lane_connector_fid'
        ].tolist()
        lane_connector_ids = list(map(str, ids))

        return [self._get_lane_connector(lane_connector_id) for lane_connector_id in lane_connector_ids]

    def _get_proximity_map_object(self, patch: geom.Polygon, layer: SemanticMapLayer) -> List[MapObject]:
        """
        Gets nearby lanes within the given patch.
        :param patch: The area to be checked.
        :param layer: desired layer to check.
        :return: A list of map objects.
        """
        layer_df = self._get_vector_map_layer(layer)
        map_object_ids = layer_df[layer_df['geometry'].intersects(patch)]['fid']

        return [self.get_map_object(map_object_id, layer) for map_object_id in map_object_ids]

    def _get_lane(self, lane_id: str) -> Lane:
        """
        Gets the lane with the given lane id.
        :param lane_id: Desired unique id of a lane that should be extracted.
        :return: Lane object.
        """
        return (
            NuPlanLane(
                lane_id,
                self._get_vector_map_layer(SemanticMapLayer.LANE),
                self._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR),
                self._get_vector_map_layer(SemanticMapLayer.BASELINE_PATHS),
                self._get_vector_map_layer(SemanticMapLayer.BOUNDARIES),
                self._get_vector_map_layer(SemanticMapLayer.STOP_LINE),
                self._load_vector_map_layer(self._LANE_CONNECTOR_POLYGON_LAYER),
                self,
            )
            if int(lane_id) in self._get_vector_map_layer(SemanticMapLayer.LANE)["lane_fid"].tolist()
            else None
        )

    def _get_lane_connector(self, lane_connector_id: str) -> LaneConnector:
        """
        Gets the lane connector with the given lane_connector_id.
        :param lane_connector_id: Desired unique id of a lane connector that should be extracted.
        :return: LaneConnector object.
        """
        return (
            NuPlanLaneConnector(
                lane_connector_id,
                self._get_vector_map_layer(SemanticMapLayer.LANE),
                self._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR),
                self._get_vector_map_layer(SemanticMapLayer.BASELINE_PATHS),
                self._get_vector_map_layer(SemanticMapLayer.BOUNDARIES),
                self._get_vector_map_layer(SemanticMapLayer.STOP_LINE),
                self._load_vector_map_layer(self._LANE_CONNECTOR_POLYGON_LAYER),
                self,
            )
            if lane_connector_id in self._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR)["fid"].tolist()
            else None
        )

    def _get_roadblock(self, roadblock_id: str) -> RoadBlockGraphEdgeMapObject:
        """
        Gets the roadblock with the given roadblock_id.
        :param roadblock_id: Desired unique id of a roadblock that should be extracted.
        :return: RoadBlock object.
        """
        return (
            NuPlanRoadBlock(
                roadblock_id,
                self._get_vector_map_layer(SemanticMapLayer.LANE),
                self._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR),
                self._get_vector_map_layer(SemanticMapLayer.BASELINE_PATHS),
                self._get_vector_map_layer(SemanticMapLayer.BOUNDARIES),
                self._get_vector_map_layer(SemanticMapLayer.ROADBLOCK),
                self._get_vector_map_layer(SemanticMapLayer.ROADBLOCK_CONNECTOR),
                self._get_vector_map_layer(SemanticMapLayer.STOP_LINE),
                self._get_vector_map_layer(SemanticMapLayer.INTERSECTION),
                self._load_vector_map_layer(self._LANE_CONNECTOR_POLYGON_LAYER),
                self,
            )
            if roadblock_id in self._get_vector_map_layer(SemanticMapLayer.ROADBLOCK)["fid"].tolist()
            else None
        )

    def _get_roadblock_connector(self, roadblock_connector_id: str) -> RoadBlockGraphEdgeMapObject:
        """
        Gets the roadblock connector with the given roadblock_connector_id.
        :param roadblock_connector_id: Desired unique id of a roadblock connector that should be extracted.
        :return: RoadBlockConnector object.
        """
        return (
            NuPlanRoadBlockConnector(
                roadblock_connector_id,
                self._get_vector_map_layer(SemanticMapLayer.LANE),
                self._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR),
                self._get_vector_map_layer(SemanticMapLayer.BASELINE_PATHS),
                self._get_vector_map_layer(SemanticMapLayer.BOUNDARIES),
                self._get_vector_map_layer(SemanticMapLayer.ROADBLOCK),
                self._get_vector_map_layer(SemanticMapLayer.ROADBLOCK_CONNECTOR),
                self._get_vector_map_layer(SemanticMapLayer.STOP_LINE),
                self._get_vector_map_layer(SemanticMapLayer.INTERSECTION),
                self._load_vector_map_layer(self._LANE_CONNECTOR_POLYGON_LAYER),
                self,
            )
            if roadblock_connector_id
            in self._get_vector_map_layer(SemanticMapLayer.ROADBLOCK_CONNECTOR)["fid"].tolist()
            else None
        )

    def _initialize_drivable_area(self) -> None:
        """
        Drivable area is considered as the union of road_segments, intersections and generic_drivable_areas.
        Hence, the three layers has to be joined to cover all drivable areas.
        """
        road_segments = self._load_vector_map_layer('road_segments')
        intersections = self._load_vector_map_layer('intersections')
        generic_drivable_areas = self._load_vector_map_layer('generic_drivable_areas')
        car_parks = self._load_vector_map_layer('carpark_areas')
        self._vector_map['drivable_area'] = pd.concat(
            [road_segments, intersections, generic_drivable_areas, car_parks]
        ).dropna(axis=1, how='any')

    def _get_stop_line(self, stop_line_id: str) -> StopLine:
        """
        Gets the stop line with the given stop_line_id.
        :param stop_line_id: desired unique id of a stop line that should be extracted.
        :return: NuPlanStopLine object.
        """
        return (
            NuPlanStopLine(stop_line_id, self._get_vector_map_layer(SemanticMapLayer.STOP_LINE))
            if stop_line_id in self._get_vector_map_layer(SemanticMapLayer.STOP_LINE)["fid"].tolist()
            else None
        )

    def _get_crosswalk(self, crosswalk_id: str) -> NuPlanPolygonMapObject:
        """
        Gets the stop line with the given crosswalk_id.
        :param crosswalk_id: desired unique id of a stop line that should be extracted.
        :return: NuPlanStopLine object.
        """
        return (
            NuPlanPolygonMapObject(crosswalk_id, self._get_vector_map_layer(SemanticMapLayer.CROSSWALK))
            if crosswalk_id in self._get_vector_map_layer(SemanticMapLayer.CROSSWALK)["fid"].tolist()
            else None
        )

    def _get_intersection(self, intersection_id: str) -> Intersection:
        """
        Gets the stop line with the given stop_line_id.
        :param intersection_id: desired unique id of a stop line that should be extracted.
        :return: NuPlanStopLine object.
        """
        return (
            NuPlanIntersection(intersection_id, self._get_vector_map_layer(SemanticMapLayer.INTERSECTION))
            if intersection_id in self._get_vector_map_layer(SemanticMapLayer.INTERSECTION)["fid"].tolist()
            else None
        )

    def _get_walkway(self, walkway_id: str) -> NuPlanPolygonMapObject:
        """
        Gets the walkway with the given walkway_id.
        :param walkway_id: desired unique id of a walkway that should be extracted.
        :return: NuPlanPolygonMapObject object.
        """
        return (
            NuPlanPolygonMapObject(walkway_id, self._get_vector_map_layer(SemanticMapLayer.WALKWAYS))
            if walkway_id in self._get_vector_map_layer(SemanticMapLayer.WALKWAYS)["fid"].tolist()
            else None
        )

    def _get_carpark_area(self, carpark_area_id: str) -> NuPlanPolygonMapObject:
        """
        Gets the car park area with the given car_park_area_id.
        :param carpark_area_id: desired unique id of a car park that should be extracted.
        :return: NuPlanPolygonMapObject object.
        """
        return (
            NuPlanPolygonMapObject(carpark_area_id, self._get_vector_map_layer(SemanticMapLayer.CARPARK_AREA))
            if carpark_area_id in self._get_vector_map_layer(SemanticMapLayer.CARPARK_AREA)["fid"].tolist()
            else None
        )
