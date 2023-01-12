import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.gridspec as gridspec
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Polygon

from nuplan.database.maps_db.map_api import NuPlanMapWrapper

# Define a map geometry type for polygons and lines.
Geometry = Union[Polygon, LineString]


class NuPlanMapExplorer:
    """Helper class to explore the nuPlan map data."""

    def __init__(self, map_api: NuPlanMapWrapper, color_map: Optional[Dict[str, str]] = None) -> None:
        """
        Constructor.
        :param map_api: A NuPlanMapWrapper instance.
        :param color_map: Color Map for each segment.
        """
        self.map_api = map_api

        if color_map is None:
            self.color_map = dict(
                generic_drivable_areas='#a6cee3',
                road_segments='#1f78b4',
                lanes_polygons='#b2df8a',
                ped_crossings='#fb9a99',
                walkways='#e31a1c',
                carpark_areas='#ff7f00',
                traffic_lights='#7e772e',
                intersections='#703642',
                lane_group_connectors='#cab2d6',
                stop_polygons='#800080',
                speed_bumps='#DC7633',
                lane_connectors='#6a3d9a',
                lane_groups_polygons='#85929E',
                boundaries='#839192',
                crosswalks='#F6DDCC',
            )
        else:
            self.color_map = color_map

    def render_map_mask(
        self,
        patch_box: Tuple[float, float, float, float],
        patch_angle: float,
        layer_names: List[str],
        output_size: Tuple[int, int],
        figsize: Tuple[int, int],
        n_row: int = 2,
    ) -> Tuple[Figure, List[Axes]]:
        """
        Render map mask of the patch specified by patch_box and patch_angle.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_names: A list of layer names to be rendered.
        :param output_size: Size of the output mask (h, w).
        :param figsize: Size of the figure.
        :param n_row: Number of rows with plots.
        :return: The matplotlib figure and a list of axes of the rendered layers.
        """
        map_dims = self.map_api.get_map_dimension()
        if output_size is None:
            output_size = (int(map_dims[1]), int(map_dims[0]))

        map_mask = self.get_map_mask(patch_box, patch_angle, layer_names, output_size)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, output_size[1])
        ax.set_ylim(0, output_size[0])

        n_col = map_mask.shape[0]
        gs = gridspec.GridSpec(n_row, n_col)
        gs.update(wspace=0.025, hspace=0.05)
        for i in range(len(map_mask)):
            r = i // n_col
            c = i - r * n_col
            subax = plt.subplot(gs[r, c])
            subax.imshow(map_mask[i], origin='lower')
            subax.text(output_size[0] * 0.5, output_size[1] * 1.1, layer_names[i])
            subax.grid(False)

        return fig, fig.axes

    def render_layers(
        self, layer_names: List[str], alpha: float, tokens: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Render a list of layers.
        :param layer_names: A list of layer names.
        :param alpha: The opacity of each layer.
        :param tokens: Dict of tokens for each layer in layer_name.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1 / self.map_api.get_map_aspect_ratio()])

        with warnings.catch_warnings():
            # Suppress ShapelyDeprecationWarning.
            warnings.filterwarnings("ignore")

            xmin, ymin = float('inf'), float('inf')
            xmax, ymax = float('-inf'), float('-inf')
            for layer_name in layer_names:
                if tokens is None:
                    bounds = self.map_api.get_bounds(layer_name)
                else:
                    bounds = self.map_api.get_bounds(layer_name, tokens[layer_name])
                xmin = min(xmin, bounds[0])
                ymin = min(ymin, bounds[1])
                xmax = max(xmax, bounds[2])
                ymax = max(ymax, bounds[3])

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            layer_names = list(set(layer_names))
            for layer_name in layer_names:
                if tokens is None:
                    self._render_layer(ax, layer_name, alpha)
                else:
                    self._render_layer(ax, layer_name, alpha, tokens[layer_name])
            ax.legend()

            return fig, ax

    def _render_layer(self, ax: Axes, layer_name: str, alpha: float, tokens: Optional[List[str]] = None) -> None:
        """
        Wrapper method that renders individual layers on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: The list of tokens of layer to render.
        """
        if layer_name in self.map_api.vector_polygon_layers:
            self._render_polygon_layer(ax, layer_name, alpha, tokens)
        elif (layer_name in self.map_api.vector_line_layers) or (layer_name in self.map_api.vector_point_layers):
            self._render_line_layer(ax, layer_name, alpha, tokens)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _render_polygon_layer(
        self, ax: Axes, layer_name: str, alpha: float, tokens: Optional[List[str]] = None
    ) -> None:
        """
        Renders an individual polygon layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: The list of tokens of layer to render.
        """
        if layer_name in self.map_api.vector_layers:
            records = self.map_api.load_vector_layer(layer_name)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

        for i in range(len(records)):
            polygons = records['geometry'][i]

            if tokens is not None:
                fid = records['fid'][i]
                if fid not in tokens:
                    continue

            xs, ys = polygons.exterior.xy
            ax.fill(xs, ys, alpha=alpha, fc=self.color_map[layer_name], ec='none')

    def _render_line_layer(self, ax: Axes, layer_name: str, alpha: float, tokens: Optional[List[str]] = None) -> None:
        """
        Renders an individual line layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: List of tokens of layer to render.
        """
        if layer_name in self.map_api.vector_layers:
            records = self.map_api.load_vector_layer(layer_name)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

        first_time = True
        for i in range(len(records)):
            line = records['geometry'][i]

            if tokens is not None:
                fid = records['fid'][i]
                if fid not in tokens:
                    continue

            if first_time:
                label = layer_name
                first_time = False
            else:
                label = None
            if line.is_empty:  # Skip lines without nodes.
                continue
            xs, ys = line.xy
            if layer_name in self.map_api.vector_point_layers:
                # Draws an circle at the position of physical traffic light.
                ax.add_patch(Circle((xs[0], ys[0]), color=self.color_map[layer_name], label=label))
            else:
                ax.plot(xs, ys, color=self.color_map[layer_name], alpha=alpha, label=label)

    def map_geom_to_mask(
        self,
        map_geom: List[Tuple[str, List[Geometry]]],
        local_box: Tuple[float, float, float, float],
        output_size: Tuple[int, int],
    ) -> npt.NDArray[np.uint8]:
        """
        Return list of map mask layers of the specified patch.
        :param map_geom: List of layer names and their corresponding geometries.
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param output_size: Size of the output mask (h, w).
        :return: Stacked numpy array of size [c x h x w] with c channels and the same height/width as the canvas.
        """
        # Get each layer mask and stack them into a numpy tensor.
        map_mask = []
        for layer_name, layer_geom in map_geom:
            layer_mask = self._layer_geom_to_mask(layer_name, layer_geom, local_box, output_size)
            if layer_mask is not None:
                map_mask.append(layer_mask)

        return np.array(map_mask)

    def get_map_mask(
        self,
        patch_box: Tuple[float, float, float, float],
        patch_angle: float,
        layer_names: List[str],
        output_size: Tuple[int, int],
    ) -> npt.NDArray[np.uint8]:
        """
        Returns list of map mask layers of the specified patch.
        :param patch_box: Patch box defined as [x_center, y_center, height, width]. If None, returns the entire map.
        :param patch_angle: Patch orientation in degrees. North-facing corresponds to 0.
        :param layer_names: A list of layer names to be extracted.
        :param output_size: Size of the output mask (h, w).
        :return: Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas.
        """
        # For some combination of parameters, we need to know the size of the current map.
        map_geom = self.get_map_geom(patch_box, patch_angle, layer_names)

        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        map_mask = self.map_geom_to_mask(map_geom, local_box, output_size)
        assert np.all(map_mask.shape[1:] == output_size)

        return map_mask

    def get_map_geom(
        self, patch_box: Tuple[float, float, float, float], patch_angle: float, layer_names: List[str]
    ) -> List[Tuple[str, List[Geometry]]]:
        """
        Returns a list of geometries in the specified patch_box.
        These are unscaled, but aligned with the patch angle.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
                            North-facing corresponds to 0.
        :param layer_names: A list of layer names to be extracted.
        :return: List of layer names and their corresponding geometries.
        """
        # Get each layer name and geometry and store them in a list.
        map_geom = []
        for layer_name in layer_names:
            layer_geom = self._get_layer_geom(patch_box, patch_angle, layer_name)
            if layer_geom is None:
                continue
            map_geom.append((layer_name, layer_geom))

        return map_geom

    def _layer_geom_to_mask(
        self,
        layer_name: str,
        layer_geom: List[Geometry],
        local_box: Tuple[float, float, float, float],
        output_size: Tuple[int, int],
    ) -> npt.NDArray[np.uint8]:
        """
        Wrapper method that gets the mask for each layer's geometries.
        :param layer_name: The name of the layer for which we get the masks.
        :param layer_geom: List of the geometries of the layer specified in layer_name.
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param output_size: Size of the output mask (h, w).
        :return: Binary map mask patch in a canvas size.
        """
        if layer_name in self.map_api.vector_polygon_layers:
            return self._polygon_geom_to_mask(layer_geom, local_box, output_size)
        elif layer_name in self.map_api.vector_line_layers:
            return self._line_geom_to_mask(layer_geom, local_box, layer_name, output_size)  # type: ignore
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _polygon_geom_to_mask(
        self, layer_geom: List[LineString], local_box: Tuple[float, float, float, float], output_size: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """
        Convert polygon inside patch to binary mask and return the map patch.
        :param layer_geom: list of polygons for each map layer.
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param output_size: Size of the output mask (h, w).
        :return: Binary map mask patch with the size canvas_size.
        """
        patch_x, patch_y, patch_h, patch_w = local_box
        patch = self.map_api.get_patch_coord(local_box)

        output_h = output_size[0]
        output_w = output_size[1]
        scale_height = output_h / patch_h
        scale_width = output_w / patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        map_mask = np.zeros(output_size, np.uint8)  # type: ignore

        for polygon in layer_geom:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_polygon = affinity.scale(new_polygon, xfact=scale_width, yfact=scale_height, origin=(0, 0))

                if new_polygon.geom_type == 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                map_mask = self.mask_for_polygons(new_polygon, map_mask)

        return map_mask

    def _line_geom_to_mask(
        self,
        layer_geom: List[LineString],
        local_box: Tuple[float, float, float, float],
        layer_name: str,
        output_size: Tuple[int, int],
    ) -> Optional[npt.NDArray[np.uint8]]:
        """
        Convert line inside patch to binary mask and return the map patch.
        :param layer_geom: list of LineStrings for each map layer.
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param layer_name: name of map layer to be converted to binary map mask patch.
        :param output_size: Size of the output mask (h, w).
        :return: Binary map mask patch in a canvas size.
        """
        patch_x, patch_y, patch_h, patch_w = local_box
        patch = self.map_api.get_patch_coord(local_box)

        output_h = output_size[0]
        output_w = output_size[1]
        scale_height = output_h / patch_h
        scale_width = output_w / patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        map_mask = np.zeros(output_size, np.uint8)  # type: ignore

        if layer_name == 'traffic_light':
            return None

        for line in layer_geom:
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))

                map_mask = self.mask_for_lines(new_line, map_mask)
        return map_mask

    def _get_layer_geom(
        self, patch_box: Tuple[float, float, float, float], patch_angle: float, layer_name: str
    ) -> List[Geometry]:
        """
        Wrapper method that gets the geometries for each layer.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_name: Name of map layer to be converted to binary map mask patch.
        :return: List of geometries for the given layer.
        """
        if layer_name in self.map_api.vector_polygon_layers:
            return self.map_api.get_layer_polygon(patch_box, patch_angle, layer_name)  # type: ignore
        elif layer_name in self.map_api.vector_line_layers:
            return self.map_api.get_layer_line(patch_box, patch_angle, layer_name)  # type: ignore
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def get_nearby_roads(self, x: float, y: float) -> Dict[str, List[str]]:
        """
        Gets the possible next roads from a point of interest.
        Returns road_segment, road_block and lane.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :return: Dictionary of layer_name - tokens pairs.
        """
        # Filter out irrelevant layers.
        road_layers = ['lanes_polygons', 'road_segments']
        layers_tokens = self.map_api.layers_on_point(x, y, road_layers)

        assert layers_tokens is not None, 'Error: No suitable layer in the specified point location!'

        # Get all records that overlap with the bounding box of the selected road.
        xmin, ymin = float('inf'), float('inf')
        xmax, ymax = float('-inf'), float('-inf')
        for road_layer in road_layers:
            bounds = self.map_api.get_bounds(road_layer, layers_tokens[road_layer])
            xmin = min(xmin, bounds[0])
            ymin = min(ymin, bounds[1])
            xmax = max(xmax, bounds[2])
            ymax = max(ymax, bounds[3])

        box_coords = [xmin, ymin, xmax, ymax]
        intersect = self.map_api.get_records_in_patch(box_coords, road_layers, mode='intersect')

        return intersect  # type: ignore

    def render_nearby_roads(self, x: float, y: float, alpha: float = 0.5) -> Tuple[Figure, Axes]:
        """
        Renders the possible next roads from a point of interest.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :param alpha: The opacity of each layer that gets rendered.
        """
        nearby_roads = self.get_nearby_roads(x, y)
        layer_names = []
        for layer_name, layer_tokens in nearby_roads.items():
            if len(layer_tokens) > 0:
                layer_names.append(layer_name)

        # Render them.
        fig, ax = self.render_layers(layer_names, alpha, tokens=nearby_roads)

        # Render current location with an x.
        ax.plot(x, y, 'x', markersize=12, color='red')

        return fig, ax

    @staticmethod
    def mask_for_lines(lines: LineString, mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Convert a Shapely LineString back to an image mask ndarray.
        :param lines: List of shapely LineStrings to be converted to a numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray line mask.
        """
        if lines.geom_type == 'MultiLineString':
            for line in lines:
                coords = np.array(line.coords, np.int32)  # type: ignore
                coords = coords.reshape((-1, 2))
                cv2.polylines(mask, [coords], False, 1, 2)
        else:
            coords = np.array(lines.coords, np.int32)
            coords = coords.reshape((-1, 2))
            cv2.polylines(mask, [coords], False, 1, 2)

        return mask

    @staticmethod
    def mask_for_polygons(polygons: MultiPolygon, mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Convert a polygon or multipolygon list to an image mask ndarray.
        :param polygons: List of Shapely polygons to be converted to numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray polygon mask.
        """
        if not polygons:
            return mask

        def int_coords(x: Any) -> npt.NDArray[np.int32]:
            """
            Function to round and convert to int.
            :param x: Input data, in any form that can be converted to an array.
            :return: The converted array-like int.
            """
            return np.array(x).round().astype(np.int32)

        exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]
        interiors = [int_coords(pi.coords) for poly in polygons.geoms for pi in poly.interiors]
        cv2.fillPoly(mask, exteriors, 1)
        cv2.fillPoly(mask, interiors, 0)

        return mask
