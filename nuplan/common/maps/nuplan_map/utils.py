from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as geom
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.maps_datatypes import LaneSegmentCoords, RasterLayer, VectorLayer
from nuplan.database.maps_db.layer import MapLayer


def raster_layer_from_map_layer(map_layer: MapLayer) -> RasterLayer:
    """
    Convert MapDB's MapLayer to the generic RasterLayer
    :param map_layer: input MapLayer object
    :return: output RasterLayer object
    """
    return RasterLayer(map_layer.data, map_layer.precision, map_layer.transform_matrix)


def lane_segment_coords_from_lane_segment_vector(coords: List[List[List[float]]]) -> LaneSegmentCoords:
    """
    Convert lane segment coords [N, 2, 2] to nuPlan LaneSegmentCoords
    :param coords: lane segment coordinates in vector form
    :return: lane segment coordinates as LaneSegmentCoords
    """
    return LaneSegmentCoords(
        [(Point2D(start[0], start[1]), Point2D(end[0], end[1])) for start, end in coords])


def is_in_type(x: float, y: float, vector_layer: VectorLayer) -> bool:
    """
    :param x: [m] floating point x-coordinate in global frame
    :param y: [m] floating point y-coordinate in global frame
    :param vector_layer: vector layer to be searched through
    :return True iff position [x, y] is in any entry of type, False if it is not
    """
    assert vector_layer is not None, "type can not be None!"

    in_polygon = vector_layer.contains(geom.Point(x, y))

    return any(in_polygon.values)


def get_all_elements_with_fid(elements: gpd.geodataframe.GeoDataFrame, row_key: str, desired_key: str) \
        -> gpd.geodataframe.GeoDataFrame:
    """
    Extract all matching elements. Note, if no matching desired_key is found and empty list is returned
    :param elements: data frame from MapsDb
    :param row_key: key to extract from a row
    :param desired_key: key which is compared with the value of row_key entry
    :return: a subset of the original GeoDataFrame containing the matching key
    """
    elements = elements[elements.notna()]  # Filter out missing values
    matching_rows = elements.loc[elements[row_key] == str(desired_key)]
    if matching_rows.empty:
        matching_rows = elements.loc[elements[row_key] == int(desired_key)]

    return matching_rows


def get_element_with_fid(elements: gpd.geodataframe.GeoDataFrame, row_key: str, desired_key: str) -> pd.Series:
    """
    Extract a matching element
    :param elements: data frame from MapsDb
    :param row_key: key to extract from a row
    :param desired_key: key which is compared with the value of row_key entry
    :return row from GeoDataFrame
    """
    matching_rows = get_all_elements_with_fid(elements, row_key, desired_key)
    assert len(matching_rows) > 0, f"Could not find the desired key = {desired_key}"
    assert len(matching_rows) == 1, f"{len(matching_rows)} matching keys found. Expected to only find one." \
                                    "Try using get_all_elements_with_fid"
    return matching_rows.iloc[0]


def compute_baseline_path_heading(baseline_path: geom.linestring.LineString) -> List[float]:
    """
    Compute the heading of each coordinate to its successor coordinate. The last coordinate will have the same heading
    as the second last coordinate.
    :param baseline_path: baseline path as a shapely LineString
    :return: a list of headings associated to each starting coordinate
    """

    coords = np.asarray(baseline_path.coords)
    vectors = np.diff(coords, axis=0)  # type: ignore
    angles = np.arctan2(vectors.T[1], vectors.T[0])
    angles = np.append(angles, angles[-1])  # type: ignore  # pad end with duplicate heading

    assert len(angles) == len(coords), "Calculated heading must have the same length as input coordinates"

    return list(angles)


def compute_curvature(point1: geom.Point, point2: geom.Point, point3: geom.Point) -> float:
    """
    Estimate signed curvature along the three points
    :param point1: First point of a circle
    :param point2: Second point of a circle
    :param point3: Third point of a circle
    :return signed curvature of the three points
    """
    # points_utm is a 3-by-2 array, containing the easting and northing coordinates of 3 points
    # Compute distance to each point
    a = point1.distance(point2)
    b = point2.distance(point3)
    c = point3.distance(point1)

    # Compute inverse radius of circle using surface of triangle (for which Heron's formula is used)
    surface_2 = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))

    if surface_2 < 1e-6:
        # In this case the points are almost aligned in a lane
        return 0.0

    assert surface_2 >= 0
    k = np.sqrt(surface_2) / 4  # Heron's formula for triangle's surface
    den = a * b * c  # Denumerator; make sure there is no division by zero.
    curvature = 4 * k / den if not np.isclose(den, 0.0) else 0.0

    # The curvature is unsigned, in order to extract sign, the third point is checked wrt to point1-point2 line
    position = np.sign((point2.x - point1.x) * (point3.y - point1.y) - (point2.y - point1.y) * (point3.x - point1.x))

    return float(position * curvature)


def extract_discrete_baseline(baseline_path: geom.LineString) -> List[StateSE2]:
    """
    Returns a discretized baseline composed of StateSE2 as nodes
    :param baseline_path: the baseline of interest
    :returns: baseline path as a list of waypoints represented by StateSE2
    """
    assert baseline_path.length > 0.0, "The length of the path has to be greater than 0!"

    headings = compute_baseline_path_heading(baseline_path)
    x_coords, y_coords = baseline_path.coords.xy
    return [StateSE2(x, y, heading) for x, y, heading in zip(x_coords, y_coords, headings)]


def estimate_curvature_along_path(path: geom.LineString, arc_length: float,
                                  distance_for_curvature_estimation: float) -> float:
    """
    Estimate curvature along a path at arc_length from origin
    :param path: LineString creating a continuous path
    :param arc_length: [m] distance from origin of the path
    :param distance_for_curvature_estimation: [m] the distance used to construct 3 points
    :return estimated curvature at point arc_length
    """
    assert 0 <= arc_length <= path.length
    # Extract 3 points from a path
    if path.length < 2.0 * distance_for_curvature_estimation:
        # In this case the arch_length is too short
        first_arch_length = 0.0
        second_arc_length = path.length / 2.0
        third_arc_length = path.length
    elif arc_length - distance_for_curvature_estimation < 0.0:
        # In this case the arch_length is too close to origin
        first_arch_length = 0.0
        second_arc_length = distance_for_curvature_estimation
        third_arc_length = 2.0 * distance_for_curvature_estimation
    elif arc_length + distance_for_curvature_estimation > path.length:
        # In this case the arch_length is too close to end of the path
        first_arch_length = path.length - 2.0 * distance_for_curvature_estimation
        second_arc_length = path.length - distance_for_curvature_estimation
        third_arc_length = path.length
    else:  # In this case the arc_length lands along the path
        first_arch_length = arc_length - distance_for_curvature_estimation
        second_arc_length = arc_length
        third_arc_length = arc_length + distance_for_curvature_estimation

    first_arch_position = path.interpolate(first_arch_length)
    second_arch_position = path.interpolate(second_arc_length)
    third_arch_position = path.interpolate(third_arc_length)
    return compute_curvature(first_arch_position, second_arch_position, third_arch_position)
