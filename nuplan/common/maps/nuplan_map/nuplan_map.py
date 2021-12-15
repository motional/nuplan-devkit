from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import shapely.geometry as geom
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map_objects import Crosswalk, Intersection, Lane, LaneConnector, StopLine
from nuplan.common.maps.maps_datatypes import LaneSegmentConnections, LaneSegmentCoords, LaneSegmentMetaData, \
    RasterLayer, RasterMap, SemanticMapLayer, StopLineType, VectorLayer
from nuplan.common.maps.nuplan_map.crosswalk import NuPlanCrosswalk
from nuplan.common.maps.nuplan_map.intersection import NuPlanIntersection
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.nuplan_map.stop_line import NuPlanStopLine
from nuplan.common.maps.nuplan_map.utils import is_in_type, lane_segment_coords_from_lane_segment_vector, \
    raster_layer_from_map_layer
from nuplan.database.maps_db.imapsdb import IMapsDB
from nuplan.database.maps_db.layer import MapLayer
from nuplan.database.maps_db.utils import build_lane_segments_from_blps, connect_blp_predecessor, connect_blp_successor
from nuplan.database.nuplan_db.utils import get_candidates


class NuPlanMap(AbstractMap):

    def __init__(self, maps_db: IMapsDB, map_name: str) -> None:
        """
        Initializes the map class.

        :param maps_db: MapsDB instance
        :param map_name: Name of the map
        """
        self._maps_db = maps_db
        self._vector_map: Dict[str, VectorLayer] = defaultdict(VectorLayer)
        self._raster_map: Dict[str, RasterLayer] = defaultdict(RasterLayer)
        self._map_name = map_name

        self._map_object_getter: Dict[SemanticMapLayer, Callable[[str], MapObject]] = \
            {SemanticMapLayer.LANE: self._get_lane,
             SemanticMapLayer.LANE_CONNECTOR: self._get_lane_connector,
             SemanticMapLayer.STOP_LINE: self._get_stop_line,
             SemanticMapLayer.CROSSWALK: self._get_crosswalk,
             SemanticMapLayer.INTERSECTION: self._get_intersection}

        self._vector_layer_mapping = {
            SemanticMapLayer.LANE: 'lanes_polygons',
            SemanticMapLayer.INTERSECTION: 'intersections',
            SemanticMapLayer.STOP_LINE: 'stop_polygons',
            SemanticMapLayer.CROSSWALK: 'crosswalks',
            SemanticMapLayer.SPEED_BUMP: 'speed_bumps',
            SemanticMapLayer.DRIVABLE_AREA: 'drivable_area',
            SemanticMapLayer.LANE_CONNECTOR: 'lane_connectors',
            SemanticMapLayer.BASELINE_PATHS: 'baseline_paths',
        }
        self._raster_layer_mapping = {
            SemanticMapLayer.DRIVABLE_AREA: 'drivable_area',
        }

    @property
    def map_name(self) -> str:
        """ Inherited, see superclass. """
        return self._map_name

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """ Inherited, see superclass. """
        return list(self._map_object_getter.keys())

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        """ Inherited, see superclass. """
        return list(self._raster_layer_mapping.keys())

    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        """ Inherited, see superclass. """
        layer_id = self._semantic_raster_layer_map(layer)
        return self._load_raster_layer(layer_id)

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        """ Inherited, see superclass. """
        raster_map = RasterMap(layers=defaultdict(RasterLayer))
        for layer in layers:
            raster_map.layers[layer] = self.get_raster_map_layer(layer)
        return raster_map

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        """ Inherited, see superclass. """
        if layer == SemanticMapLayer.TURN_STOP:
            stop_lines = self._get_vector_map_layer(SemanticMapLayer.STOP_LINE)
            in_stop_line = stop_lines.loc[stop_lines.contains(geom.Point(point.x, point.y))]
            return any(in_stop_line.loc[in_stop_line["stop_polygon_type_fid"] == StopLineType.TURN_STOP.value].values)

        return bool(is_in_type(point.x, point.y, self._get_vector_map_layer(layer)))

    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        """ Inherited, see superclass. """
        try:
            return self._get_all_map_objects(point, layer)
        except KeyError:
            raise ValueError(f"Object representation for layer: {layer.name} is unavailable")

    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
        """ Inherited, see superclass. """
        map_objects = self.get_all_map_objects(point, layer)
        if len(map_objects) > 1:
            raise AssertionError(f"{len(map_objects)} map objects found. Expected only one. "
                                 "Try using get_all_map_objects()")

        if len(map_objects) == 0:
            return None

        return map_objects[0]

    def get_proximal_map_objects(self, point: Point2D, radius: float, layers: List[SemanticMapLayer]) \
            -> Dict[SemanticMapLayer, List[MapObject]]:
        """ Inherited, see superclass. """
        x_min, x_max = point.x - radius, point.x + radius
        y_min, y_max = point.y - radius, point.y + radius
        patch = geom.box(x_min, y_min, x_max, y_max)

        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            ValueError(f"Object representation for layer(s): {unsupported_layers} is unavailable")

        object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

        for layer in layers:
            object_map[layer] = self._get_proximity_map_object(patch, layer)

        return object_map

    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> MapObject:
        """ Inherited, see superclass. """
        try:
            return self._map_object_getter[layer](object_id)
        except KeyError:
            raise ValueError(f"Object representation for layer: {layer.name} is unavailable")

    def get_distance_to_nearest_map_object(self, point: Point2D, layer: SemanticMapLayer) \
            -> Tuple[Optional[str], Optional[float]]:
        """ Inherited from superclass """
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

    def get_neighbor_vector_map(self, point: Point2D, radius: float) -> \
            Tuple[LaneSegmentCoords, LaneSegmentConnections, LaneSegmentMetaData]:
        """ Inherited from superclass """
        ls_coords: List[List[List[float]]] = []
        ls_conns: List[Tuple[int, int]] = []
        ls_meta: List[List[Any]] = []
        cross_blp_connection: Dict[str, List[int]] = dict()

        blps_gdf = self._load_vector_map_layer('baseline_paths')
        lane_poly_gdf = self._load_vector_map_layer('lanes_polygons')
        intersections_gdf = self._load_vector_map_layer('intersections')
        lane_connectors_gdf = self._load_vector_map_layer('lane_connectors')
        lane_groups_gdf = self._load_vector_map_layer('lane_groups_polygons')

        position = (point.x, point.y)
        xrange, yrange = (-radius, radius), (-radius, radius)

        if (blps_gdf is None) or (lane_poly_gdf is None) or (intersections_gdf is None) or \
                (lane_connectors_gdf is None) or (lane_groups_gdf is None):
            return lane_segment_coords_from_lane_segment_vector(ls_coords), \
                LaneSegmentConnections(ls_conns), \
                LaneSegmentMetaData(ls_meta)

        # data enhancement
        blps_in_lanes = blps_gdf[blps_gdf['lane_fid'].notna()]
        blps_in_intersections = blps_gdf[blps_gdf['lane_connector_fid'].notna()]

        # enhance blps_in_lanes
        lane_group_info = lane_poly_gdf[['lane_fid', 'lane_group_fid']]
        blps_in_lanes = blps_in_lanes.merge(lane_group_info, on='lane_fid', how='outer')

        # enhance blps_in_intersections
        lane_connectors_gdf['lane_connector_fid'] = lane_connectors_gdf['fid']
        lane_conns_info = lane_connectors_gdf[['lane_connector_fid',
                                               'intersection_fid', 'exit_lane_fid', 'entry_lane_fid']]
        # Convert the exit_fid field of both data frames to the same dtype for merging.
        lane_conns_info = lane_conns_info.astype({'lane_connector_fid': int})
        blps_in_intersections = blps_in_intersections.astype({'lane_connector_fid': int})
        blps_in_intersections = blps_in_intersections.merge(lane_conns_info, on='lane_connector_fid', how='outer')

        # enhance blps_connection info
        lane_blps_info = blps_in_lanes[['fid', 'lane_fid']]
        from_blps_info = lane_blps_info.rename(columns={'fid': 'from_blp', 'lane_fid': 'exit_lane_fid'})
        to_blps_info = lane_blps_info.rename(columns={'fid': 'to_blp', 'lane_fid': 'entry_lane_fid'})
        blps_in_intersections = blps_in_intersections.merge(from_blps_info, on='exit_lane_fid', how='inner')
        blps_in_intersections = blps_in_intersections.merge(to_blps_info, on='entry_lane_fid', how='inner')

        # Select in-range blps
        if radius > 0:
            candidate_lane_groups, candidate_intersections = get_candidates(
                position, xrange, yrange, lane_groups_gdf, intersections_gdf)
            candidate_blps_in_lanes = blps_in_lanes[blps_in_lanes['lane_group_fid'].isin(
                candidate_lane_groups['fid'].astype(int))]
            candidate_blps_in_intersections = blps_in_intersections[blps_in_intersections['intersection_fid'].isin(
                candidate_intersections['fid'].astype(int))]
        else:
            candidate_blps_in_lanes = blps_in_lanes
            candidate_blps_in_intersections = blps_in_intersections

        # generate lane_segments from blps in lanes
        build_lane_segments_from_blps(candidate_blps_in_lanes, ls_coords, ls_conns, cross_blp_connection)
        # generate lane_segments from blps in intersections
        build_lane_segments_from_blps(candidate_blps_in_intersections, ls_coords,
                                      ls_conns, cross_blp_connection)

        # generate connections between blps
        for blp_id, blp_info in cross_blp_connection.items():
            # Add predecessors
            connect_blp_predecessor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_conns)
            # Add successors
            connect_blp_successor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_conns)

        return lane_segment_coords_from_lane_segment_vector(ls_coords), \
            LaneSegmentConnections(ls_conns), \
            LaneSegmentMetaData(ls_meta)

    def _semantic_vector_layer_map(self, layer: SemanticMapLayer) -> str:
        """
        Mapping from SemanticMapLayer int to MapsDB internal representation of vector layers
        :param layer: The querired semantic map layer
        :return: A internal layer name as a string
        @raise ValueError if the requested layer does not exist for MapsDBMap
        """
        try:
            return self._vector_layer_mapping[layer]
        except KeyError:
            raise ValueError("Unknown layer: {}".format(layer.name))

    def _semantic_raster_layer_map(self, layer: SemanticMapLayer) -> str:
        """
        Mapping from SemanticMapLayer int to MapsDB internal representation of raster layers
        :param layer: The querired semantic map layer
        :return: A internal layer name as a string
        @raise ValueError if the requested layer does not exist for MapsDBMap
        """
        try:
            return self._raster_layer_mapping[layer]
        except KeyError:
            raise ValueError("Unknown layer: {}".format(layer.name))

    def _get_vector_map_layer(self, layer: SemanticMapLayer) -> VectorLayer:
        """ Inherited, see superclass. """
        layer_id = self._semantic_vector_layer_map(layer)
        return self._load_vector_map_layer(layer_id)

    def _load_raster_layer(self, layer_name: str) -> RasterLayer:
        """
        Load and cache raster layers
        :layer_name: the name of the vector layer to be loaded
        :return: the loaded RasterLayer
        """
        if layer_name not in self._raster_map:
            map_layer: MapLayer = self._maps_db.load_layer(self._map_name, layer_name)
            self._raster_map[layer_name] = raster_layer_from_map_layer(map_layer)

        return self._raster_map[layer_name]

    def _load_vector_map_layer(self, layer_name: str) -> VectorLayer:
        """
        Load and cache vector layers
        :layer_name: the name of the vector layer to be loaded
        :return: the loaded VectorLayer
        """
        if layer_name not in self._vector_map:
            if layer_name == 'drivable_area':
                self._intialize_drivable_area()
            else:
                self._vector_map[layer_name] = self._maps_db.load_vector_layer(self._map_name, layer_name)

        return self._vector_map[layer_name]

    def _get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        """
        Gets a list of lanes where its polygon overlaps the queried point
        :param point: [m] x, y coordinates in global frame
        :return: a list of lanes. An empty list if no lanes were found
        """
        if layer == SemanticMapLayer.LANE_CONNECTOR:
            return self._get_all_lane_connectors(point)
        else:
            layer_df = self._get_vector_map_layer(layer)
            ids = layer_df.loc[layer_df.contains(geom.Point(point.x, point.y))]['fid'].tolist()
            return [self._map_object_getter[layer](map_object_id) for map_object_id in ids]

    def _get_all_lane_connectors(self, point: Point2D) -> List[LaneConnector]:
        """
        Gets a list of lane connectors where its polygon overlaps the queried point
        :param point: [m] x, y coordinates in global frame
        :return: a list of lane connectors. An empty list if no lane connectors were found
        """
        lane_connectors_df = self._load_vector_map_layer('gen_lane_connectors_scaled_width_polygons')
        ids = lane_connectors_df.loc[lane_connectors_df.contains(geom.Point(point.x,
                                                                            point.y))]['lane_connector_fid'].tolist()
        lane_connector_ids = list(map(str, ids))
        return [self._get_lane_connector(lane_connector_id) for lane_connector_id in lane_connector_ids]

    def _get_proximity_map_object(self, patch: geom.Polygon, layer: SemanticMapLayer) -> List[MapObject]:
        """
        Gets nearby lanes within the given patch.
        :param patch: The area to be checked.
        :param layer: desired layer to check
        :return: A list of map objects.
        """
        layer_df = self._get_vector_map_layer(layer)
        map_object_ids = layer_df[layer_df['geometry'].intersects(patch)]["fid"]
        return [self._map_object_getter[layer](map_object_id) for map_object_id in map_object_ids]

    def _get_lane(self, lane_id: str) -> Lane:
        """
        Gets the lane with the given lane id.
        :param lane_id: Desired unique id of a lane that should be extracted.
        :return: Lane object.
        """
        return NuPlanLane(lane_id,
                          self._get_vector_map_layer(SemanticMapLayer.LANE),
                          self._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR),
                          self._get_vector_map_layer(SemanticMapLayer.BASELINE_PATHS)) \
            if int(lane_id) in self._get_vector_map_layer(SemanticMapLayer.LANE)["lane_fid"].tolist() else None

    def _get_lane_connector(self, lane_connector_id: str) -> LaneConnector:
        """
        Gets the lane connector with the given lane_connector_id.
        :param lane_connector_id: Desired unique id of a lane connector that should be extracted.
        :return: LaneConnector object.
        """
        return NuPlanLaneConnector(lane_connector_id,
                                   self._get_vector_map_layer(SemanticMapLayer.LANE),
                                   self._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR),
                                   self._get_vector_map_layer(SemanticMapLayer.BASELINE_PATHS)) \
            if lane_connector_id in self._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR)["fid"].tolist() \
            else None

    def _intialize_drivable_area(self) -> None:
        """
        Drivable area is considered as the union of road_segments, intersections and generic_drivable_areas.
        Hence, the three layers has to be joined to cover all drivable areas.
        """
        road_segments = self._load_vector_map_layer('road_segments')
        intersections = self._load_vector_map_layer('intersections')
        generic_drivable_areas = self._load_vector_map_layer('generic_drivable_areas')
        self._vector_map['drivable_area'] = road_segments.append(
            intersections.append(
                generic_drivable_areas)).dropna(axis=1, how='any')

    def _get_stop_line(self, stop_line_id: str) -> StopLine:
        """
        Gets the stop line with the given stop_line_id.
        :param stop_line_id: desired unique id of a stop line that should be extracted.
        :return: NuPlanStopLine object.
        """
        return NuPlanStopLine(stop_line_id, self._get_vector_map_layer(SemanticMapLayer.STOP_LINE)) \
            if stop_line_id in self._get_vector_map_layer(SemanticMapLayer.STOP_LINE)["fid"].tolist() else None

    def _get_crosswalk(self, crosswalk_id: str) -> Crosswalk:
        """
        Gets the stop line with the given crosswalk_id.
        :param crosswalk_id: desired unique id of a stop line that should be extracted.
        :return: NuPlanStopLine object.
        """
        return NuPlanCrosswalk(crosswalk_id, self._get_vector_map_layer(SemanticMapLayer.CROSSWALK)) \
            if crosswalk_id in self._get_vector_map_layer(SemanticMapLayer.CROSSWALK)["fid"].tolist() else None

    def _get_intersection(self, intersection_id: str) -> Intersection:
        """
        Gets the stop line with the given stop_line_id.
        :param intersection_id: desired unique id of a stop line that should be extracted.
        :return: NuPlanStopLine object.
        """
        return NuPlanIntersection(intersection_id, self._get_vector_map_layer(SemanticMapLayer.INTERSECTION)) \
            if intersection_id in self._get_vector_map_layer(SemanticMapLayer.INTERSECTION)["fid"].tolist() else None
