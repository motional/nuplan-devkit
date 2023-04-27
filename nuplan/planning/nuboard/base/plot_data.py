from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
from bokeh.document import Document
from bokeh.models import Button, ColumnDataSource, GlyphRenderer, HoverTool, LayoutDOM, Legend, Slider, Title
from bokeh.plotting.figure import Figure

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.transform import translate_longitudinally
from nuplan.common.maps.abstract_map_objects import LaneConnector
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.nuboard.style import (
    simulation_map_layer_color,
    simulation_tile_agent_style,
    simulation_tile_style,
    simulation_tile_trajectory_style,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.utils.serialization.to_scene import tracked_object_types


class BokehAgentStates(NamedTuple):
    """Agent states in bokeh."""

    xs: List[List[List[List[float]]]]  # [m], [[list of [[Polygon connected corners in x]]]]
    ys: List[List[List[List[float]]]]  # [m], [[list of [[Polygon connected corners in y]]]]
    agent_type: List[str]  # A list of agent's category
    track_id: List[Union[int, float]]  # A list of agent's track id
    track_token: List[str]  # A list of agent's track token
    center_xs: List[float]  # [m], a list of center in x.
    center_ys: List[float]  # [m], a list of center in y.
    velocity_xs: List[float]  # [m/s], A list of velocity in x (body frame).
    velocity_ys: List[float]  # [m/s], A list of velocity in y (body frame).
    speeds: List[float]  # [m/s], A list of speed.
    headings: List[float]  # [m], a list of headings


@dataclass(frozen=True)
class MapPoint:
    """A dataclass to render map polygons in scenario."""

    point_2d: List[List[Point2D]] = field(default_factory=list)  # A list of a list of 2D points

    @property
    def polygon_xs(self) -> List[List[List[List[float]]]]:
        """Return a list of xs from point 2d to render polygons."""
        polygon_xs = []
        for points in self.point_2d:
            xs = []
            for point in points:
                xs.append(point.x)
            polygon_xs.append([[xs]])
        return polygon_xs

    @property
    def polygon_ys(self) -> List[List[List[List[float]]]]:
        """Return a list of ys from point 2d to render polygons."""
        polygon_ys = []
        for points in self.point_2d:
            ys = []
            for point in points:
                ys.append(point.y)
            polygon_ys.append([[ys]])
        return polygon_ys

    @property
    def line_xs(self) -> List[List[float]]:
        """Return a list of xs from point 2d to render lines."""
        line_xs = []
        for points in self.point_2d:
            xs = []
            for point in points:
                xs.append(point.x)
            line_xs.append(xs)
        return line_xs

    @property
    def line_ys(self) -> List[List[float]]:
        """Return a list of ys from point 2d to render lines."""
        line_ys = []
        for points in self.point_2d:
            ys = []
            for point in points:
                ys.append(point.y)
            line_ys.append(ys)
        return line_ys


@dataclass(frozen=True)
class TrafficLightMapLine(MapPoint):
    """Line plot data in traffic light map."""

    line_colors: List[str] = field(default_factory=list)  # A list of color hex codes.
    line_color_alphas: List[float] = field(default_factory=list)  # A list of color alphas.


@dataclass
class BaseScenarioPlot(abc.ABC):
    """Base class for scenario plot classes."""

    # Threading condition to synchronize data source production & consumption:
    data_source_condition: Optional[threading.Condition] = field(default=None, init=False)

    # Threading event that will be set when rendering starts and cleared when rendering ends.
    render_event: Optional[threading.Event] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize threading properties."""
        if not self.data_source_condition:
            self.data_source_condition = threading.Condition(threading.Lock())
        if not self.render_event:
            self.render_event = threading.Event()


@dataclass
class TrafficLightPlot(BaseScenarioPlot):
    """A dataclass for traffic light plot."""

    data_sources: Dict[int, ColumnDataSource] = field(default_factory=dict)  # A dict of data sources for each frame
    plot: Optional[GlyphRenderer] = None  # A bokeh glyph element

    def update_plot(self, main_figure: Figure, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param frame_index: Frame index.
        :param doc: The Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return

        self.render_event.set()  # type: ignore

        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                data_sources = dict(self.data_sources[frame_index].data)
                if self.plot is None:
                    self.plot = main_figure.multi_line(
                        xs="xs",
                        ys="ys",
                        line_color="line_colors",
                        line_alpha="line_color_alphas",
                        line_width=3.0,
                        line_dash="dashed",
                        source=data_sources,
                    )
                else:
                    self.plot.data_source.data = data_sources

                self.render_event.clear()  # type: ignore

            # Defer updating the main_figure to the next tick to be compatible with multi-threading approach
            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(
        self, scenario: AbstractScenario, history: SimulationHistory, lane_connectors: Dict[str, LaneConnector]
    ) -> None:
        """
        Update traffic light status datasource of each frame.
        :param scenario: Scenario traffic light status information.
        :param history: SimulationHistory time-series data.
        :param lane_connectors: Lane connectors.
        """
        if not self.data_source_condition:
            return

        with self.data_source_condition:
            for frame_index in range(len(history.data)):
                traffic_light_status = history.data[frame_index].traffic_light_status

                traffic_light_map_line = TrafficLightMapLine(point_2d=[], line_colors=[], line_color_alphas=[])
                lane_connector_colors = simulation_map_layer_color[SemanticMapLayer.LANE_CONNECTOR]
                for traffic_light in traffic_light_status:
                    lane_connector = lane_connectors.get(str(traffic_light.lane_connector_id), None)

                    if lane_connector is not None:
                        path = lane_connector.baseline_path.discrete_path
                        points = [Point2D(x=pose.x, y=pose.y) for pose in path]
                        traffic_light_map_line.line_colors.append(traffic_light.status.name)
                        traffic_light_map_line.line_color_alphas.append(lane_connector_colors["line_color_alpha"])
                        traffic_light_map_line.point_2d.append(points)

                line_source = ColumnDataSource(
                    dict(
                        xs=traffic_light_map_line.line_xs,
                        ys=traffic_light_map_line.line_ys,
                        line_colors=traffic_light_map_line.line_colors,
                        line_color_alphas=traffic_light_map_line.line_color_alphas,
                    )
                )
                self.data_sources[frame_index] = line_source
                self.data_source_condition.notify()


@dataclass
class EgoStatePlot(BaseScenarioPlot):
    """A dataclass for ego state plot."""

    vehicle_parameters: VehicleParameters  # Ego vehicle parameters
    data_sources: Dict[int, ColumnDataSource] = field(default_factory=dict)  # A dict of data sources for each frame
    init_state: bool = True  # True to indicate it is in init state
    plot: Optional[GlyphRenderer] = None  # A bokeh glyph element

    def update_plot(self, main_figure: Figure, radius: float, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param radius: Figure radius.
        :param frame_index: Frame index.
        :param doc: Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return

        self.render_event.set()  # type: ignore

        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            data_sources = dict(self.data_sources[frame_index].data)
            center_x = data_sources["center_x"][0]
            center_y = data_sources["center_y"][0]

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                if self.plot is None:
                    self.plot = main_figure.multi_polygons(
                        xs="xs",
                        ys="ys",
                        fill_color=simulation_tile_agent_style["ego"]["fill_color"],
                        fill_alpha=simulation_tile_agent_style["ego"]["fill_alpha"],
                        line_color=simulation_tile_agent_style["ego"]["line_color"],
                        line_width=simulation_tile_agent_style["ego"]["line_width"],
                        source=data_sources,
                    )
                    ego_hover = HoverTool(
                        renderers=[self.plot],
                        tooltips=[
                            ("center_x [m]", "@center_x{0.2f}"),
                            ("center_y [m]", "@center_y{0.2f}"),
                            ("velocity_x [m/s]", "@velocity_x{0.2f}"),
                            ("velocity_y [m/s]", "@velocity_y{0.2f}"),
                            ("speed [m/s", "@speed{0.2f}"),
                            ("acceleration_x [m/s^2]", "@acceleration_x{0.2f}"),
                            ("acceleration_y [m/s^2]", "@acceleration_y{0.2f}"),
                            ("acceleration [m/s^2]", "@acceleration{0.2f}"),
                            ("heading [rad]", "@heading{0.2f}"),
                            ("steering_angle [rad]", "@steering_angle{0.2f}"),
                            ("yaw_rate [rad/s]", "@yaw_rate{0.2f}"),
                            ("type", "Ego"),
                        ],
                    )
                    main_figure.add_tools(ego_hover)
                else:
                    self.plot.data_source.data = data_sources

                # This will (re)center the view around the ego, actually making the plot visible
                if self.init_state:
                    main_figure.x_range.start = center_x - radius / 2
                    main_figure.x_range.end = center_x + radius / 2
                    main_figure.y_range.start = center_y - radius / 2
                    main_figure.y_range.end = center_y + radius / 2
                    self.init_state = False
                else:
                    x_radius = main_figure.x_range.end - main_figure.x_range.start
                    y_radius = main_figure.y_range.end - main_figure.y_range.start
                    main_figure.x_range.start = center_x - x_radius / 2
                    main_figure.x_range.end = center_x + x_radius / 2
                    main_figure.y_range.start = center_y - y_radius / 2
                    main_figure.y_range.end = center_y + y_radius / 2

                self.render_event.clear()  # type: ignore

            # Defer updating the main_figure to the next tick to be compatible with multi-threading approach
            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update ego_pose state data sources.
        :param history: SimulationHistory time-series data.
        """
        if not self.data_source_condition:
            return

        with self.data_source_condition:
            for frame_index, sample in enumerate(history.data):
                ego_pose = sample.ego_state.car_footprint
                dynamic_car_state = sample.ego_state.dynamic_car_state
                ego_corners = ego_pose.all_corners()

                corner_xs = [corner.x for corner in ego_corners]
                corner_ys = [corner.y for corner in ego_corners]

                # Connect to the first point
                corner_xs.append(corner_xs[0])
                corner_ys.append(corner_ys[0])
                source = ColumnDataSource(
                    dict(
                        center_x=[ego_pose.center.x],
                        center_y=[ego_pose.center.y],
                        velocity_x=[dynamic_car_state.rear_axle_velocity_2d.x],
                        velocity_y=[dynamic_car_state.rear_axle_velocity_2d.y],
                        speed=[dynamic_car_state.speed],
                        acceleration_x=[dynamic_car_state.rear_axle_acceleration_2d.x],
                        acceleration_y=[dynamic_car_state.rear_axle_acceleration_2d.y],
                        acceleration=[dynamic_car_state.acceleration],
                        heading=[ego_pose.center.heading],
                        steering_angle=[sample.ego_state.tire_steering_angle],
                        yaw_rate=[sample.ego_state.dynamic_car_state.angular_velocity],
                        xs=[[[corner_xs]]],
                        ys=[[[corner_ys]]],
                    )
                )
                self.data_sources[frame_index] = source
                self.data_source_condition.notify()


@dataclass
class EgoStateTrajectoryPlot(BaseScenarioPlot):
    """A dataclass for ego state trajectory plot."""

    data_sources: Dict[int, ColumnDataSource] = field(default_factory=dict)  # A dict of data sources for each frame
    plot: Optional[GlyphRenderer] = None  # A bokeh glyph element

    def update_plot(self, main_figure: Figure, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param frame_index: Frame index.
        :param doc: Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return

        self.render_event.set()  # type: ignore

        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            data_sources = dict(self.data_sources[frame_index].data)

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                if self.plot is None:
                    self.plot = main_figure.line(
                        x="xs",
                        y="ys",
                        line_color=simulation_tile_trajectory_style["ego"]["line_color"],
                        line_width=simulation_tile_trajectory_style["ego"]["line_width"],
                        line_alpha=simulation_tile_trajectory_style["ego"]["line_alpha"],
                        source=data_sources,
                    )
                else:
                    self.plot.data_source.data = data_sources

                self.render_event.clear()  # type: ignore

            # Defer updating the main_figure to the next tick to be compatible with multi-threading approach
            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update ego_pose trajectory data sources.
        :param history: SimulationHistory time-series data.
        """
        if not self.data_source_condition:
            return

        with self.data_source_condition:
            for frame_index, sample in enumerate(history.data):
                trajectory = sample.trajectory.get_sampled_trajectory()

                x_coords = []
                y_coords = []
                for state in trajectory:
                    x_coords.append(state.center.x)
                    y_coords.append(state.center.y)

                source = ColumnDataSource(dict(xs=x_coords, ys=y_coords))
                self.data_sources[frame_index] = source
                self.data_source_condition.notify()


@dataclass
class AgentStatePlot(BaseScenarioPlot):
    """A dataclass for agent state plot."""

    data_sources: Dict[int, Dict[str, ColumnDataSource]] = field(default_factory=dict)  # A dict of data for each frame
    plots: Dict[str, GlyphRenderer] = field(default_factory=dict)  # A dict of plots for each type
    track_id_history: Optional[Dict[str, int]] = None  # Track id history

    def __post_init__(self) -> None:
        """Initialize track id history."""
        super().__post_init__()

        if not self.track_id_history:
            self.track_id_history = {}

    def _get_track_id(self, track_id: str) -> Union[int, float]:
        """
        Get a number representation for track ids.
        :param track_id: Agent track id.
        :return A number representation for a track id.
        """
        if track_id == "null" or not self.track_id_history:
            return np.nan

        number_track_id = self.track_id_history.get(track_id, None)
        if not number_track_id:
            self.track_id_history[track_id] = len(self.track_id_history)
            number_track_id = len(self.track_id_history)

        return number_track_id

    def update_plot(self, main_figure: Figure, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param frame_index: Frame index.
        :param doc: Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return

        self.render_event.set()  # type: ignore

        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                data_sources = self.data_sources.get(frame_index, None)
                if not data_sources:
                    return

                for category, data_source in data_sources.items():
                    plot = self.plots.get(category, None)
                    data = dict(data_source.data)
                    if plot is None:
                        agent_color = simulation_tile_agent_style.get(category)
                        self.plots[category] = main_figure.multi_polygons(
                            xs="xs",
                            ys="ys",
                            fill_color=agent_color["fill_color"],
                            fill_alpha=agent_color["fill_alpha"],
                            line_color=agent_color["line_color"],
                            line_width=agent_color["line_width"],
                            source=data,
                        )
                        agent_hover = HoverTool(
                            renderers=[self.plots[category]],
                            tooltips=[
                                ("center_x [m]", "@center_xs{0.2f}"),
                                ("center_y [m]", "@center_ys{0.2f}"),
                                ("velocity_x [m/s]", "@velocity_xs{0.2f}"),
                                ("velocity_y [m/s]", "@velocity_ys{0.2f}"),
                                ("speed [m/s]", "@speeds{0.2f}"),
                                ("heading [rad]", "@headings{0.2f}"),
                                ("type", "@agent_type"),
                                ("track token", "@track_token"),
                            ],
                        )
                        main_figure.add_tools(agent_hover)
                    else:
                        self.plots[category].data_source.data = data

                self.render_event.clear()  # type: ignore

            # Defer updating the main_figure to the next tick to be compatible with multi-threading approach
            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update agents data sources.
        :param history: SimulationHistory time-series data.
        """
        if not self.data_source_condition:
            return

        with self.data_source_condition:
            for frame_index, sample in enumerate(history.data):
                if not isinstance(sample.observation, DetectionsTracks):
                    continue

                tracked_objects = sample.observation.tracked_objects
                frame_dict = {}
                for tracked_object_type_name, tracked_object_type in tracked_object_types.items():
                    corner_xs = []
                    corner_ys = []
                    track_ids = []
                    track_tokens = []
                    agent_types = []
                    center_xs = []
                    center_ys = []
                    velocity_xs = []
                    velocity_ys = []
                    speeds = []
                    headings = []

                    for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type):
                        agent_corners = tracked_object.box.all_corners()
                        corners_x = [corner.x for corner in agent_corners]
                        corners_y = [corner.y for corner in agent_corners]
                        corners_x.append(corners_x[0])
                        corners_y.append(corners_y[0])
                        corner_xs.append([[corners_x]])
                        corner_ys.append([[corners_y]])
                        center_xs.append(tracked_object.center.x)
                        center_ys.append(tracked_object.center.y)
                        velocity_xs.append(tracked_object.velocity.x)
                        velocity_ys.append(tracked_object.velocity.y)
                        speeds.append(tracked_object.velocity.magnitude())
                        headings.append(tracked_object.center.heading)
                        agent_types.append(tracked_object_type.fullname)
                        track_ids.append(self._get_track_id(tracked_object.track_token))
                        track_tokens.append(tracked_object.track_token)

                    agent_states = BokehAgentStates(
                        xs=corner_xs,
                        ys=corner_ys,
                        track_id=track_ids,
                        track_token=track_tokens,
                        agent_type=agent_types,
                        center_xs=center_xs,
                        center_ys=center_ys,
                        velocity_xs=velocity_xs,
                        velocity_ys=velocity_ys,
                        speeds=speeds,
                        headings=headings,
                    )

                    frame_dict[tracked_object_type_name] = ColumnDataSource(agent_states._asdict())

                self.data_sources[frame_index] = frame_dict
                self.data_source_condition.notify()


@dataclass
class AgentStateHeadingPlot(BaseScenarioPlot):
    """A dataclass for agent state heading plot."""

    data_sources: Dict[int, Dict[str, ColumnDataSource]] = field(default_factory=dict)  # A dict of data for each frame
    plots: Dict[str, GlyphRenderer] = field(default_factory=dict)  # A dict of plots for each type

    def update_plot(self, main_figure: Figure, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param frame_index: Frame index.
        :param doc: Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return

        self.render_event.set()  # type: ignore

        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                data_sources = self.data_sources.get(frame_index, None)
                if not data_sources:
                    return

                for category, data_source in data_sources.items():
                    plot = self.plots.get(category, None)
                    data = dict(data_source.data)
                    if plot is None:
                        agent_color = simulation_tile_agent_style.get(category)
                        self.plots[category] = main_figure.multi_line(
                            xs="trajectory_x",
                            ys="trajectory_y",
                            line_color=agent_color["line_color"],
                            line_width=agent_color["line_width"],
                            source=data,
                        )
                    else:
                        self.plots[category].data_source.data = data

                self.render_event.clear()  # type: ignore

            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update agent heading data sources.
        :param history: SimulationHistory time-series data.
        """
        if not self.data_source_condition:
            return

        with self.data_source_condition:
            for frame_index, sample in enumerate(history.data):
                if not isinstance(sample.observation, DetectionsTracks):
                    continue

                tracked_objects = sample.observation.tracked_objects
                frame_dict: Dict[str, Any] = {}
                for tracked_object_type_name, tracked_object_type in tracked_object_types.items():
                    trajectory_xs = []
                    trajectory_ys = []
                    for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type):
                        object_box = tracked_object.box
                        agent_trajectory = translate_longitudinally(
                            object_box.center, distance=object_box.length / 2 + 1
                        )
                        trajectory_xs.append([object_box.center.x, agent_trajectory.x])
                        trajectory_ys.append([object_box.center.y, agent_trajectory.y])

                    trajectories = ColumnDataSource(
                        dict(
                            trajectory_x=trajectory_xs,
                            trajectory_y=trajectory_ys,
                        )
                    )
                    frame_dict[tracked_object_type_name] = trajectories

                self.data_sources[frame_index] = frame_dict
                self.data_source_condition.notify()


@dataclass
class SimulationFigure:
    """Simulation figure data."""

    # Required simulation data
    planner_name: str  # Planner name
    scenario: AbstractScenario  # Scenario
    simulation_history: SimulationHistory  # SimulationHistory
    vehicle_parameters: VehicleParameters  # Ego parameters

    # Rendering objects
    figure: Figure  # Bokeh figure
    file_path_index: int  # Experiment file index
    slider: Slider  # Bokeh slider to this figure
    video_button: Button  # Bokeh video button to this figure
    first_button: Button  # Bokeh button to go to the first frame
    prev_button: Button  # Bokeh button to go back one frame
    play_button: Button  # Bokeh button to automatically advance frames
    next_button: Button  # Bokeh button to advance one frame
    last_button: Button  # Bokeh button to go to the last frame
    figure_title_name: str  # Figure title name
    x_y_coordinate_title: Title  # Title renderer for x and y coordinate
    time_us: Optional[List[int]] = None  # Timestamp in microsecond
    mission_goal_plot: Optional[GlyphRenderer] = None  # Mission goal plot
    expert_trajectory_plot: Optional[GlyphRenderer] = None  # Expert trajectory plot
    legend_state: bool = False  # Legend states
    map_polygon_plots: Dict[str, GlyphRenderer] = field(default_factory=dict)  # Polygon plots for map layers
    map_line_plots: Dict[str, GlyphRenderer] = field(default_factory=dict)  # Line plots for map layers
    traffic_light_plot: Optional[TrafficLightPlot] = None  # Traffic light plot
    ego_state_plot: Optional[EgoStatePlot] = None  # Ego state plot
    ego_state_trajectory_plot: Optional[EgoStateTrajectoryPlot] = None  # Ego state trajectory plot
    agent_state_plot: Optional[AgentStatePlot] = None  # Agent state plot
    agent_state_heading_plot: Optional[AgentStateHeadingPlot] = None  # Agent state heading plot

    # Optional simulation data
    lane_connectors: Optional[Dict[str, LaneConnector]] = None  # Lane connector id: lane connector

    # Dataclass
    glyph_names_from_checkbox_group: Optional[Dict[str, str]] = None  # Correct glyph names from checkbox groups

    def __post_init__(self) -> None:
        """Initialize all plots and data sources."""
        if self.lane_connectors is None:
            self.lane_connectors = {}

        if self.time_us is None:
            self.time_us = []

        if self.traffic_light_plot is None:
            self.traffic_light_plot = TrafficLightPlot()

        if self.ego_state_plot is None:
            self.ego_state_plot = EgoStatePlot(vehicle_parameters=self.vehicle_parameters)

        if self.ego_state_trajectory_plot is None:
            self.ego_state_trajectory_plot = EgoStateTrajectoryPlot()

        if self.agent_state_plot is None:
            self.agent_state_plot = AgentStatePlot()

        if self.agent_state_heading_plot is None:
            self.agent_state_heading_plot = AgentStateHeadingPlot()

    def is_rendering(self) -> bool:
        """:return: true if at least one plot is currently rendering a frame request."""
        plots = [
            self.traffic_light_plot,
            self.ego_state_plot,
            self.ego_state_trajectory_plot,
            self.agent_state_plot,
            self.agent_state_heading_plot,
        ]
        return any(plot.render_event.is_set() if plot.render_event else False for plot in plots if plot)

    def figure_title_name_with_timestamp(self, frame_index: int) -> str:
        """
        Return figure title with a timestamp.
        :param frame_index: Frame index.
        """
        if self.time_us:
            return f"{self.figure_title_name} (Frame: {frame_index}, Time_us: {self.time_us[frame_index]})"
        else:
            return self.figure_title_name

    def copy_datasources(self, other: SimulationFigure) -> None:
        """
        Copy data sources from another simulation figure.
        :param other: Another SimulationFigure object.
        """
        self.time_us = other.time_us
        self.scenario = other.scenario
        self.simulation_history = other.simulation_history
        self.lane_connectors = other.lane_connectors
        self.traffic_light_plot.data_sources = other.traffic_light_plot.data_sources  # type: ignore
        self.ego_state_plot.data_sources = other.ego_state_plot.data_sources  # type: ignore
        self.ego_state_trajectory_plot.data_sources = other.ego_state_trajectory_plot.data_sources  # type: ignore
        self.agent_state_plot.data_sources = other.agent_state_plot.data_sources  # type: ignore
        self.agent_state_heading_plot.data_sources = other.agent_state_heading_plot.data_sources  # type: ignore

    def update_data_sources(self) -> None:
        """
        Update data sources in a multi-threading manner to speed up loading and initialization in
        scenario rendering.
        """
        if len(self.simulation_history.data) == 0:
            raise ValueError("SimulationHistory cannot be empty!")

        # Update slider steps
        self.slider.end = len(self.simulation_history.data) - 1

        # Update time_us
        self.time_us = [sample.ego_state.time_us for sample in self.simulation_history.data]

        # Update plot data sources
        for plot in [
            self.ego_state_plot,
            self.ego_state_trajectory_plot,
            self.agent_state_plot,
            self.agent_state_heading_plot,
        ]:
            if plot:
                t = threading.Thread(target=plot.update_data_sources, args=(self.simulation_history,), daemon=True)
                t.start()

    def update_map_dependent_data_sources(self) -> None:
        """
        Update data sources in a multi-threading manner to speed up loading and initialization in
        scenario rendering.
        """
        if len(self.simulation_history.data) == 0:
            raise ValueError("SimulationHistory cannot be empty!")

        if self.lane_connectors is not None and len(self.lane_connectors):
            if not self.traffic_light_plot:
                return

            thread = threading.Thread(
                target=self.traffic_light_plot.update_data_sources,
                args=(
                    self.scenario,
                    self.simulation_history,
                    self.lane_connectors,
                ),
                daemon=True,
            )
            thread.start()

    def render_mission_goal(self, mission_goal_state: StateSE2) -> None:
        """
        Render the mission goal.
        :param mission_goal_state: Mission goal state.
        """
        source = ColumnDataSource(
            dict(xs=[mission_goal_state.x], ys=[mission_goal_state.y], heading=[mission_goal_state.heading])
        )
        self.mission_goal_plot = self.figure.rect(
            x="xs",
            y="ys",
            height=self.vehicle_parameters.height,
            width=self.vehicle_parameters.length,
            angle="heading",
            fill_alpha=simulation_tile_style["mission_goal_alpha"],
            color=simulation_tile_style["mission_goal_color"],
            line_width=simulation_tile_style["mission_goal_line_width"],
            source=source,
        )

    def render_expert_trajectory(self, expert_ego_trajectory_state: ColumnDataSource) -> None:
        """
        Render expert trajectory.
        :param expert_ego_trajectory_state: A list of trajectory states.
        """
        self.expert_trajectory_plot = self.figure.line(
            x="xs",
            y="ys",
            line_color=simulation_tile_trajectory_style["expert_ego"]["line_color"],
            line_alpha=simulation_tile_trajectory_style["expert_ego"]["line_alpha"],
            line_width=simulation_tile_trajectory_style["expert_ego"]["line_width"],
            source=expert_ego_trajectory_state,
        )

    @staticmethod
    def _update_glyph_visibility(glyphs: List[Optional[GlyphRenderer]]) -> None:
        """
        Update visibility in a list of glyphs.
        :param glyphs: A list of glyphs.
        """
        for glyph in glyphs:
            if glyph is not None:
                glyph.visible = not glyph.visible

    def get_glyph_name_from_checkbox_group(self, glyph_checkbox_group_name: str) -> str:
        """
        Get the correct glyph name of each glyph type based on the name from checkbox group.
        :param glyph_checkbox_group_name: glyph name from a checkbox group.
        :return Correct glyph name based on the glyph name from checkbox groups.
        """
        if not self.glyph_names_from_checkbox_group:
            self.glyph_names_from_checkbox_group = {
                'Vehicle': 'vehicles',
                'Pedestrian': 'pedestrians',
                'Bicycle': 'bicycles',
                'Generic': 'genericobjects',
                'Traffic Cone': 'traffic_cone',
                'Barrier': 'barrier',
                'Czone Sign': 'czone_sign',
                'Lane': SemanticMapLayer.LANE.name,
                'Intersection': SemanticMapLayer.INTERSECTION.name,
                'Stop Line': SemanticMapLayer.STOP_LINE.name,
                'Crosswalk': SemanticMapLayer.CROSSWALK.name,
                'Walkway': SemanticMapLayer.WALKWAYS.name,
                'Carpark': SemanticMapLayer.CARPARK_AREA.name,
                'RoadBlock': SemanticMapLayer.ROADBLOCK.name,
                'Lane Connector': SemanticMapLayer.LANE_CONNECTOR.name,
                'Lane Line': SemanticMapLayer.LANE.name,
            }

        name = self.glyph_names_from_checkbox_group.get(glyph_checkbox_group_name, None)
        if not name:
            raise ValueError(f"{glyph_checkbox_group_name} is not a valid glyph name!")
        return name

    def _get_trajectory_glyph_to_update(self, glyph_name: str) -> List[Optional[GlyphRenderer]]:
        """
        Get a trajectory glyph to update its visibility.
        :param glyph_name: Glyph name.
        :return A list of glyphs to be updated.
        """
        if glyph_name == 'Expert Trajectory':
            return [self.expert_trajectory_plot if self.expert_trajectory_plot is not None else None]
        elif glyph_name == 'Ego Trajectory':
            return [self.ego_state_trajectory_plot.plot if self.ego_state_trajectory_plot is not None else None]
        elif glyph_name == 'Goal':
            return [self.mission_goal_plot]
        elif glyph_name == 'Traffic Light':
            return [self.traffic_light_plot.plot if self.traffic_light_plot is not None else None]
        else:
            raise ValueError(f"{glyph_name} is not a valid trajectory name.")

    def _get_agent_glyph_to_update(self, glyph_name: str) -> List[Optional[GlyphRenderer]]:
        """
        Update an agent glyph to update its visibility.
        :param glyph_name: Glyph name.
        :return A list of glyphs to be updated.
        """
        object_type_name = self.get_glyph_name_from_checkbox_group(glyph_checkbox_group_name=glyph_name)
        return [
            self.agent_state_plot.plots.get(object_type_name, None) if self.agent_state_plot is not None else None,
            self.agent_state_heading_plot.plots.get(object_type_name, None)
            if self.agent_state_heading_plot is not None
            else None,
        ]

    def update_glyphs_visibility(self, glyph_names: Optional[List[str]] = None) -> None:
        """
        Update glyphs' visibility based on a list of glyph names.
        :param glyph_names: List of glyph names to update their visibility.
        """
        if not glyph_names:
            return
        glyphs = []
        for glyph_name in glyph_names:
            if glyph_name == 'Ego':
                glyphs += [self.ego_state_plot.plot if self.ego_state_plot is not None else None]
            elif glyph_name in ['Expert Trajectory', 'Ego Trajectory', 'Goal', 'Traffic Light']:
                glyphs += self._get_trajectory_glyph_to_update(glyph_name=glyph_name)
            elif glyph_name in ['Vehicle', 'Pedestrian', 'Bicycle', 'Generic', 'Traffic Cone', 'Barrier', 'Czone Sign']:
                glyphs += self._get_agent_glyph_to_update(glyph_name=glyph_name)
            elif glyph_name in ['Lane', 'Intersection', 'Stop Line', 'Crosswalk', 'Walkway', 'Carpark', 'RoadBlock']:
                map_polygon_name = self.get_glyph_name_from_checkbox_group(glyph_checkbox_group_name=glyph_name)
                glyphs += [self.map_polygon_plots.get(map_polygon_name, None)]
            elif glyph_name in ['Lane Connector', 'Lane Line']:
                map_line_name = self.get_glyph_name_from_checkbox_group(glyph_checkbox_group_name=glyph_name)
                glyphs += [self.map_line_plots.get(map_line_name, None)]

        self._update_glyph_visibility(glyphs=glyphs)

    def update_legend(self) -> None:
        """Update legend."""
        if self.legend_state:
            return

        if not self.agent_state_heading_plot or not self.agent_state_plot:
            return

        agent_legends = [
            (category.capitalize(), [plot, self.agent_state_heading_plot.plots[category]])
            for category, plot in self.agent_state_plot.plots.items()
        ]

        selected_map_polygon_layers = [
            SemanticMapLayer.LANE.name,
            SemanticMapLayer.INTERSECTION.name,
            SemanticMapLayer.STOP_LINE.name,
            SemanticMapLayer.CROSSWALK.name,
            SemanticMapLayer.WALKWAYS.name,
            SemanticMapLayer.CARPARK_AREA.name,
        ]
        map_polygon_legend_items = []
        for map_polygon_layer in selected_map_polygon_layers:
            map_polygon_legend_items.append(
                (map_polygon_layer.capitalize(), [self.map_polygon_plots[map_polygon_layer]])
            )
        selected_map_line_layers = [SemanticMapLayer.LANE.name, SemanticMapLayer.LANE_CONNECTOR.name]
        map_line_legend_items = []
        for map_line_layer in selected_map_line_layers:
            map_line_legend_items.append((map_line_layer.capitalize(), [self.map_line_plots[map_line_layer]]))

        if not self.ego_state_plot or not self.ego_state_trajectory_plot:
            return

        legend_items = [
            ("Ego", [self.ego_state_plot.plot]),
            ("Ego traj", [self.ego_state_trajectory_plot.plot]),
        ]
        if self.mission_goal_plot is not None:
            legend_items.append(("Goal", [self.mission_goal_plot]))

        if self.expert_trajectory_plot is not None:
            legend_items.append(("Expert traj", [self.expert_trajectory_plot]))

        legend_items += agent_legends
        legend_items += map_polygon_legend_items
        legend_items += map_line_legend_items
        if self.traffic_light_plot and self.traffic_light_plot.plot is not None:
            legend_items.append(("Traffic light", [self.traffic_light_plot.plot]))

        legend = Legend(items=legend_items)
        legend.click_policy = "hide"

        self.figure.add_layout(legend)
        self.legend_state = True
        self.figure.legend.label_text_font_size = '0.8em'


@dataclass
class SimulationData:
    """Simulation figure data."""

    planner_name: str  # Planner name
    simulation_figure: SimulationFigure  # Simulation figure data
    plot: LayoutDOM  # Figure plot
