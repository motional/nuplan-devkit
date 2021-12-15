import json
import logging
import lzma
import pathlib
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgpack
from bokeh.document.document import Document
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Glyph, HoverTool, Line, MultiLine, MultiPolygons, Slider
from bokeh.plotting import figure
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.transform_state import get_front_left_corner, get_front_right_corner, \
    get_rear_left_corner, get_rear_right_corner, translate_longitudinally
from nuplan.common.actor_state.vehicle_parameters import BoxParameters, VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import StopLine
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.planning.nuboard.base.data_class import BokehAgentStates, SimulationScenarioKey
from nuplan.planning.nuboard.style import simulation_map_layer_color, simulation_tile_style
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder

logger = logging.getLogger(__name__)


def extract_source_from_states(states: List[Dict[str, Any]]) -> ColumnDataSource:
    """ Helper function to get the xy coordinates into ColumnDataSource format from a list of states.
    :param states: List of states (containing the pose)
    :return: A ColumnDataSource object containing the xy coordinates.
    """
    x_coords = []
    y_coords = []
    for state in states:
        x_coords.append(state['pose'][0])
        y_coords.append(state['pose'][1])
    source = ColumnDataSource(dict(
        xs=x_coords,
        ys=y_coords)
    )
    return source


def _extract_serialization_type(first_file: pathlib.Path) -> str:
    """
    Deduce the serialization type
    :param first_file: serialized file
    :return: one from ["msgpack", "pickle", "json"].
    """
    msg_pack = first_file.suffixes == ['.msgpack', '.xz']
    msg_pickle = first_file.suffixes == ['.pkl', '.xz']
    msg_json = first_file.suffix == ".json"
    number_of_available_types = int(msg_pack) + int(msg_json) + int(msg_pickle)

    # We can handle only conclusive serialization type
    if number_of_available_types != 1:
        raise RuntimeError(f"Inconclusive file type: {first_file}!")

    if msg_pickle:
        return "pickle"
    elif msg_json:
        return "json"
    elif msg_pack:
        return "msgpack"
    else:
        raise RuntimeError("Unknown condition!")


def _load_data(file_name: pathlib.Path, serialization_type: str) -> Any:
    """
    Load data from file_name
    :param file_name: the name of a file which we want to deserialize
    :param serialization_type: type of serialization of the file
    :return: deserialized type
    """
    if serialization_type == "json":
        with open(str(file_name), 'r') as f:  # type: ignore
            return json.load(f)
    elif serialization_type == "msgpack":
        with lzma.open(str(file_name), "rb") as f:
            return msgpack.unpackb(f.read())
    elif serialization_type == "pickle":
        with lzma.open(str(file_name), "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown serialization type: {serialization_type}!")


class SimulationTile:

    def __init__(self,
                 doc: Document,
                 vehicle_parameters: VehicleParameters,
                 scenario_builder: AbstractScenarioBuilder,
                 period_milliseconds: int = 5000,
                 radius: float = 150.0):
        """
        Scenario simulation tile.
        :param doc: Bokeh HTML document.
        :param vehicle_parameters: Ego pose parameters.
        :param scenario_builder: Scenario builder instance.
        :param period_milliseconds: Milli seconds to update the tile.
        :param radius: Map radius.
        """

        self._doc = doc
        self._vehicle_parameters = vehicle_parameters
        self._scenario_builder = scenario_builder
        self._period_milliseconds = period_milliseconds
        self._radius = radius
        self._selected_scenario_keys: List[SimulationScenarioKey] = []
        self._maps: Dict[str, AbstractMap] = {}

    def _map_api(self, map_name: str) -> AbstractMap:

        if map_name not in self._maps:
            self._maps[map_name] = self._scenario_builder.get_map_api(map_name)

        return self._maps[map_name]

    def _init_simulations(self) -> None:
        """ Initialization of the visualization of simulation panel. """

        self._figures: List[figure] = []
        self._sliders: List[Slider] = []

        self._scenes: List[Dict[Path, Any]] = []
        self._ego_state_plots: List[Optional[Glyph]] = []
        self._ego_state_trajectory_plots: List[Optional[Glyph]] = []
        self._agent_state_plots: List[Optional[Glyph]] = []
        self._agent_state_heading_plots: List[Optional[Glyph]] = []
        self._agent_state_trajectory_plots: List[Optional[Glyph]] = []
        self._ego_state_data_sources: List[List[ColumnDataSource]] = []
        self._ego_state_trajectory_data_sources: List[List[ColumnDataSource]] = []
        self._agent_data_sources: List[List[ColumnDataSource]] = []
        self._agent_state_trajectory_data_sources: List[List[ColumnDataSource]] = []

        for figure_index in range(len(self._selected_scenario_keys)):
            self._scenes.append({})
            self._ego_state_plots.append(None)
            self._ego_state_trajectory_plots.append(None)
            self._agent_state_plots.append(None)
            self._agent_state_heading_plots.append(None)
            self._agent_state_trajectory_plots.append(None)
            self._ego_state_data_sources.append([])
            self._ego_state_trajectory_data_sources.append([])
            self._agent_data_sources.append([])
            self._agent_state_trajectory_data_sources.append([])
            simulation_figure = figure(x_range=(-self._radius, self._radius), y_range=(-self._radius, self._radius),
                                       plot_width=simulation_tile_style['figure_sizes'][0],
                                       plot_height=simulation_tile_style['figure_sizes'][1],
                                       title=f"{self._selected_scenario_keys[figure_index].planner_name}",
                                       tools=["pan", "wheel_zoom", "save", "reset"],
                                       match_aspect=True, active_scroll="wheel_zoom",
                                       margin=simulation_tile_style['figure_margins'],
                                       background_fill_color=simulation_tile_style['background_color'],
                                       )
            simulation_figure.axis.visible = False
            simulation_figure.xgrid.visible = False
            simulation_figure.ygrid.visible = False
            simulation_figure.title.text_font_size = simulation_tile_style['figure_title_text_font_size']
            simulation_figure.rect(fill_color=simulation_tile_style['planner_color'], legend_label="Ego")
            simulation_figure.rect(fill_color=simulation_tile_style['agent_color'], legend_label="Agents")
            simulation_figure.line(line_color=simulation_tile_style['planner_color'], legend_label="Planned trajectory")
            simulation_figure.line(line_color=simulation_tile_style['expert_color'], legend_label="Expert trajectory")
            simulation_figure.legend.background_fill_color = 'lightgray'
            simulation_figure.legend.label_text_font_style = 'bold'
            simulation_figure.legend.glyph_height = 25
            simulation_figure.legend.glyph_width = 25
            slider = Slider(start=0, end=1, value=0, step=1, title="Frame",
                            margin=simulation_tile_style['slider_margins'])
            slider.on_change("value", partial(self._slider_on_change, figure_index=figure_index))
            self._figures.append(simulation_figure)
            self._sliders.append(slider)

    @property
    def figures(self) -> List[figure]:
        """
        Access bokeh figures.
        :return A list of bokeh figures.
        """

        return self._figures

    @property
    def sliders(self) -> List[Slider]:
        """
        Access bokeh sliders.
        :return A list of bokeh sliders.
        """

        return self._sliders

    def _update_data_sources(self, scene: Dict[str, Any], figure_index: int, file_index: int) -> None:
        """
        Update data sources once there are new data sources.
        :param scene: A dict of scene data.
        :param figure_index: A figure index.
        :param file_index: A file index.
        """

        if file_index != 0:
            self._sliders[figure_index].end = file_index
        self._update_ego_state(ego_state=scene['ego'], figure_index=figure_index)
        self._update_ego_state_trajectory(trajectory=scene['trajectories']['ego_predicted_trajectory'],
                                          figure_index=figure_index)
        self._update_agents(observations=scene['world'], figure_index=figure_index)

    def _slider_on_change(self, attr: str, old: int, new: int, figure_index: int) -> None:
        """
        Helper function when a slider changes.
        :param attr: Attribute name.
        :param old: Old value.
        :param new: New value.
        :param figure_index: Figure index.
        """

        if new != len(self._scenes[figure_index]):
            self._render_plots(frame_index=new, figure_index=figure_index)

    def _read_files(self) -> None:
        """ Read all simulation files to memory. """

        for figure_index, simulation_scenario_key in enumerate(self._selected_scenario_keys):
            sorted_files = sorted(simulation_scenario_key.files, reverse=False)
            if len(sorted_files) == 0:
                raise RuntimeError("No files were found!")

            # Deduce the type of files
            first_file = sorted_files[0]
            serialization = _extract_serialization_type(first_file)

            if len(sorted_files) > 1:
                # Load scenes from all the available files
                for file_index, file in enumerate(sorted_files):
                    self._scenes[figure_index][file] = _load_data(file, serialization)
                    self._update_data_sources(self._scenes[figure_index][file], figure_index, file_index)
            else:
                # Load all scenes in one go
                file = first_file
                scenes = _load_data(file, serialization)
                scenes = scenes if isinstance(scenes, list) else [scenes]
                self._load_scenes(figure_index, scenes)

    def _load_scenes(self, figure_index: int, scenes: List[Dict[str, Any]]) -> None:
        """
        Load all scenes corresponding to figure_index
        :param figure_index: index of the loading figure
        :param scenes: all scenes to be loaded
        """
        for file_index, scene in enumerate(scenes):
            timestamp_us = pathlib.Path(str(scene["ego"]["timestamp_us"]))
            self._scenes[figure_index][timestamp_us] = scene
            self._update_data_sources(self._scenes[figure_index][timestamp_us], figure_index, file_index)

    @staticmethod
    def _render_simulation_layouts(figures: List[figure], sliders: List[Slider]) -> List[Any]:
        """
        Render simulation layouts.
        :param figures: A list of figures.
        :param sliders: A list of sliders.
        :return: A list of columns or rows.
        """

        grid_layouts = []
        for figure_plot, slider in zip(figures, sliders):
            grid_layouts.append(gridplot([[figure_plot], [slider]]))
        return grid_layouts

    def render_simulation_tiles(self, selected_scenario_keys: List[SimulationScenarioKey]) -> List[Any]:
        """
        Render simulation tiles.
        :param selected_scenario_keys: A list of selected scenario keys.
        :return A list of bokeh layouts.
        """

        self._selected_scenario_keys = selected_scenario_keys
        self._init_simulations()
        self._read_files()
        if len(self._scenes) > 0:
            self._render_scenario()

        layouts = self._render_simulation_layouts(self._figures, self._sliders)
        return layouts

    def _render_scenario(self) -> None:
        """ Render scenario. """

        if len(self._scenes) == 0:
            return

        for figure_index, scene in enumerate(self._scenes):
            files = list(scene.keys())

            if len(files) == 0:
                continue

            # Load the first file only.
            scene_states = scene[files[0]]
            self._render_map(figure_index=figure_index, scene=scene_states)

            expert_ego_trajectory = None
            if 'ego_expert_trajectory' in scene_states['trajectories']:
                expert_ego_trajectory = scene_states['trajectories']['ego_expert_trajectory']

            if expert_ego_trajectory is not None:
                self._render_expert_trajectory(expert_ego_trajectory=expert_ego_trajectory, figure_index=figure_index)

            self._render_plots(frame_index=0, figure_index=figure_index)

    def _render_map(self, figure_index: int, scene: Dict[str, Any]) -> None:
        """
        Render a map.
        :param figure_index: Index of the figure.
        :param scene: A dictionary of scene info.
        """

        map_name = scene['map_name']
        map_api = self._map_api(map_name)
        layer_names = [SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.LANE,
                       SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE]
        ego_pose = scene['ego']['pose']
        center = Point2D(ego_pose[0], ego_pose[1])

        nearest_vector_map = map_api.get_proximal_map_objects(center, self._radius, layer_names)
        # Filter out stop polygons in turn stop
        if SemanticMapLayer.STOP_LINE in nearest_vector_map:
            stop_polygons: List[StopLine] = nearest_vector_map[SemanticMapLayer.STOP_LINE]
            nearest_vector_map[SemanticMapLayer.STOP_LINE] = [stop_polygon for stop_polygon in stop_polygons
                                                              if stop_polygon.stop_line_type == StopLineType.TURN_STOP]

        # Draw polygons
        polygon_layer_names = \
            [(SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.LANE]),
             (SemanticMapLayer.INTERSECTION, simulation_map_layer_color[SemanticMapLayer.INTERSECTION]),
             (SemanticMapLayer.STOP_LINE, simulation_map_layer_color[SemanticMapLayer.STOP_LINE]),
             (SemanticMapLayer.CROSSWALK, simulation_map_layer_color[SemanticMapLayer.CROSSWALK])]

        polygon_xs = []
        polygon_ys = []
        fill_colors = []
        fill_color_alphas = []
        line_colors = []
        for layer_name, color in polygon_layer_names:
            layer = nearest_vector_map[layer_name]
            for map_obj in layer:
                xs = []
                ys = []
                coords = map_obj.polygon.exterior.coords
                for x, y in coords:
                    xs.append(x)
                    ys.append(y)
                fill_colors.append(color['fill_color'])
                fill_color_alphas.append(color['fill_color_alpha'])
                line_colors.append(color['line_color'])
                polygon_xs.append([[xs]])
                polygon_ys.append([[ys]])

        polygon_source = ColumnDataSource(dict(
            xs=polygon_xs,
            ys=polygon_ys,
            fill_colors=fill_colors,
            fill_color_alphas=fill_color_alphas,
            line_colors=line_colors
        )
        )
        polygon_glyph = MultiPolygons(xs="xs", ys="ys", fill_color="fill_colors", fill_alpha='fill_color_alphas',
                                      line_color="line_colors")

        # Draw lines
        line_layer_names = [(SemanticMapLayer.LANE,
                             simulation_map_layer_color[SemanticMapLayer.BASELINE_PATHS]),
                            (SemanticMapLayer.LANE_CONNECTOR,
                             simulation_map_layer_color[SemanticMapLayer.LANE_CONNECTOR])]
        line_xs = []
        line_ys = []
        line_colors = []
        line_color_alphas = []
        for layer_name, color in line_layer_names:
            layer = nearest_vector_map[layer_name]
            for map_obj in layer:
                xs = []
                ys = []
                path = map_obj.baseline_path().discrete_path()
                for pose in path:
                    xs.append(pose.x)
                    ys.append(pose.y)
                line_colors.append(color['line_color'])
                line_color_alphas.append(color['line_color_alpha'])
                line_xs.append(xs)
                line_ys.append(ys)

        line_source = ColumnDataSource(dict(
            xs=line_xs,
            ys=line_ys,
            line_colors=line_colors,
            line_color_alphas=line_color_alphas)
        )
        line_glyph = MultiLine(xs="xs", ys="ys", line_color="line_colors", line_alpha="line_color_alphas",
                               line_width=0.5, line_dash='dashed')

        figure = self._figures[figure_index]
        figure.add_glyph(polygon_source, polygon_glyph)
        figure.add_glyph(line_source, line_glyph)

    def _render_mission_goal(self, mission_goal_state: Dict[str, Any], figure_index: int) -> None:
        """
        Render the mission goal.
        :param mission_goal_state: Mission goal state.
        :param figure_index: Figure index.
        """

        pose = mission_goal_state['pose']
        source = ColumnDataSource(dict(
            xs=[pose[0]],
            ys=[pose[1]],
            heading=[pose[2]]
        ))
        self._figures[figure_index].circle_cross(
            x="xs",
            y="ys",
            size=simulation_tile_style['mission_goal_size'],
            fill_alpha=simulation_tile_style['mission_goal_alpha'],
            angle="heading",
            color=simulation_tile_style['mission_goal_color'],
            line_width=simulation_tile_style['mission_goal_line_width'],
            source=source
        )

    def _render_expert_trajectory(self, expert_ego_trajectory: Dict[str, Any], figure_index: int) -> None:
        """
        Render expert trajectory.
        :param expert_ego_trajectory: A list of trajectory states.
        :param figure_index: Figure index.
        """

        source = extract_source_from_states(expert_ego_trajectory["states"])

        glyph = Line(x="xs", y="ys", line_color=simulation_tile_style['expert_color'],
                     line_width=simulation_tile_style['expert_trajectory_line_width'])
        self._figures[figure_index].add_glyph(source, glyph)

    def _update_ego_state(self, ego_state: Dict[str, Any], figure_index: int) -> None:
        """
        Update ego state.
        :param ego_state: A dict of ego states.
        :param figure_index: Figure index.
        """

        pose = ego_state['pose']
        ego_state_se: StateSE2 = StateSE2(
            x=pose[0],
            y=pose[1],
            heading=pose[2]
        )
        ego_corners = [get_front_left_corner(ego_state_se,
                                             self._vehicle_parameters.half_length,
                                             self._vehicle_parameters.half_width),
                       get_front_right_corner(ego_state_se,
                                              self._vehicle_parameters.half_length,
                                              self._vehicle_parameters.half_width),
                       get_rear_right_corner(ego_state_se,
                                             self._vehicle_parameters.half_length,
                                             self._vehicle_parameters.half_width),
                       get_rear_left_corner(ego_state_se,
                                            self._vehicle_parameters.half_length,
                                            self._vehicle_parameters.half_width)]
        corner_xs = [corner.x for corner in ego_corners]
        corner_ys = [corner.y for corner in ego_corners]

        # Connect to the first point
        corner_xs.append(corner_xs[0])
        corner_ys.append(corner_ys[0])
        source = ColumnDataSource(
            dict(
                x=[ego_state_se.x],
                y=[ego_state_se.y],
                xs=[[[corner_xs]]],
                ys=[[[corner_ys]]])
        )

        self._ego_state_data_sources[figure_index].append(source)

    def _update_ego_state_trajectory(self, trajectory: Dict[str, Any], figure_index: int) -> None:
        """
        Render trajectory.
        :param trajectory: Trajectory to be rendered.
        :param figure_index: Figure index
        """

        source = extract_source_from_states(trajectory["states"])
        self._ego_state_trajectory_data_sources[figure_index].append(source)

    def _update_agents(self, observations: Dict[str, List[Dict[str, Any]]], figure_index: int) -> None:
        """
        Update agents.
        :param observations: A dict of a list of scenes.
        :param figure_index: Figure index.
        """

        corner_xs = []
        corner_ys = []
        track_ids = []
        agent_types = []
        trajectory_xs = []
        trajectory_ys = []
        for category, scenes in observations.items():
            for scene in scenes:
                pose = scene['box']['pose']
                sizes = scene['box']['size']
                state = StateSE2(x=pose[0], y=pose[1], heading=pose[2])
                box_param = BoxParameters(length=sizes[1], width=sizes[0])
                agent_corners = [get_front_left_corner(state, box_param.half_length, box_param.half_width),
                                 get_front_right_corner(state, box_param.half_length, box_param.half_width),
                                 get_rear_right_corner(state, box_param.half_length, box_param.half_width),
                                 get_rear_left_corner(state, box_param.half_length, box_param.half_width)]
                corners_x = [corner.x for corner in agent_corners]
                corners_y = [corner.y for corner in agent_corners]
                corners_x.append(corners_x[0])
                corners_y.append(corners_y[0])
                corner_xs.append([[corners_x]])
                corner_ys.append([[corners_y]])
                agent_trajectory = translate_longitudinally(state, distance=sizes[1] / 2 + 1)
                trajectory_xs.append([state.x, agent_trajectory.x])
                trajectory_ys.append([state.y, agent_trajectory.y])
                agent_types.append(scene['type'])
                track_ids.append(scene['id'])

        agent_states = BokehAgentStates(
            xs=corner_xs,
            ys=corner_ys,
            track_id=track_ids,
            agent_type=agent_types,
            trajectory_x=trajectory_xs,
            trajectory_y=trajectory_ys
        )
        self._agent_data_sources[figure_index].append(ColumnDataSource(agent_states._asdict()))

    def _render_plots(self, frame_index: int, figure_index: int) -> None:
        """
        Render plot with a frame index.
        :param frame_index: A frame index.
        :param figure_index: A figure index.
        """

        ego_state = self._ego_state_data_sources[figure_index][frame_index]
        ego_state = dict(ego_state.data)

        main_figure = self._figures[figure_index]
        center_x = ego_state["x"][0]
        center_y = ego_state["y"][0]

        if self._ego_state_plots[figure_index] is None:
            # Ego state initial plot.

            self._ego_state_plots[figure_index] = main_figure.multi_polygons(
                xs="xs", ys="ys",
                fill_color=simulation_tile_style['planner_color'],
                line_width=simulation_tile_style['planner_line_width'],
                source=ego_state)
        else:
            self._ego_state_plots[figure_index].data_source.data = ego_state  # type: ignore
        main_figure.x_range.start = center_x - self._radius / 2
        main_figure.x_range.end = center_x + self._radius / 2
        main_figure.y_range.start = center_y - self._radius / 2
        main_figure.y_range.end = center_y + self._radius / 2

        # Update ego pose trajectory state.
        ego_state_trajectory_data_source = self._ego_state_trajectory_data_sources[figure_index][frame_index]
        ego_state_trajectory_data_source = dict(ego_state_trajectory_data_source.data)
        if self._ego_state_trajectory_plots[figure_index] is None:
            self._ego_state_trajectory_plots[figure_index] = main_figure.line(
                x="xs",
                y="ys",
                line_color=simulation_tile_style['planner_color'],
                line_width=simulation_tile_style['planner_line_width'],
                source=ego_state_trajectory_data_source
            )
        else:
            ego_state_trajectory_plot = self._ego_state_trajectory_plots[figure_index]
            ego_state_trajectory_plot.data_source.data = ego_state_trajectory_data_source  # type: ignore

        agent_data_source = self._agent_data_sources[figure_index][frame_index]
        agent_data_source = dict(agent_data_source.data)
        if self._agent_state_plots[figure_index] is None:

            # Agent state
            self._agent_state_plots[figure_index] = main_figure.multi_polygons(
                xs="xs",
                ys="ys",
                fill_color=simulation_tile_style['agent_color'],
                line_width=simulation_tile_style['agent_line_width'],
                source=agent_data_source)
            agent_hover = HoverTool(renderers=[self._agent_state_plots[figure_index]],
                                    tooltips=[("Type", "@agent_type"), ("Track id", "@track_id")])
            main_figure.add_tools(agent_hover)
        else:
            self._agent_state_plots[figure_index].data_source.data = agent_data_source  # type: ignore

        if self._agent_state_heading_plots[figure_index] is None:

            # Agent state heading
            self._agent_state_heading_plots[figure_index] = main_figure.multi_line(
                xs="trajectory_x",
                ys="trajectory_y",
                line_color=simulation_tile_style['agent_trajectory_color'],
                line_width=simulation_tile_style['agent_trajectory_line_width'],
                source=agent_data_source)
        else:
            self._agent_state_heading_plots[figure_index].data_source.data = agent_data_source  # type: ignore
