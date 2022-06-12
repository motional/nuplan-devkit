import json
import logging
import lzma
import pathlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import msgpack
import numpy as np
from bokeh.document import without_document_lock
from bokeh.document.document import Document
from bokeh.io.export import get_screenshot_as_png
from bokeh.layouts import column, gridplot
from bokeh.models import Button, ColumnDataSource, Slider
from bokeh.plotting import figure
from selenium import webdriver
from tornado import gen
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.planning.nuboard.base.data_class import SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.base.plot_data import MapPoint, SimulationData, SimulationFigure
from nuplan.planning.nuboard.style import simulation_map_layer_color, simulation_tile_style

try:
    import chromedriver_binary
except ImportError:
    chromedriver_binary = None

logger = logging.getLogger(__name__)


def extract_source_from_states(states: List[Dict[str, Any]]) -> ColumnDataSource:
    """Helper function to get the xy coordinates into ColumnDataSource format from a list of states.
    :param states: List of states (containing the pose)
    :return: A ColumnDataSource object containing the xy coordinates.
    """
    x_coords = []
    y_coords = []
    for state in states:
        x_coords.append(state["pose"][0])
        y_coords.append(state["pose"][1])
    source = ColumnDataSource(dict(xs=x_coords, ys=y_coords))
    return source


def _extract_serialization_type(first_file: pathlib.Path) -> str:
    """
    Deduce the serialization type
    :param first_file: serialized file
    :return: one from ["msgpack", "pickle", "json"].
    """
    msg_pack = first_file.suffixes == [".msgpack", ".xz"]
    msg_pickle = first_file.suffixes == [".pkl", ".xz"]
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
        with open(str(file_name), "r") as f:
            return json.load(f)
    elif serialization_type == "msgpack":
        with lzma.open(str(file_name), "rb") as f:
            return msgpack.unpackb(f.read())
    elif serialization_type == "pickle":
        with lzma.open(
            str(file_name),
            "rb",
        ) as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown serialization type: {serialization_type}!")


class SimulationTile:
    """Scenario simulation tile for visualization."""

    def __init__(
        self,
        doc: Document,
        experiment_file_data: ExperimentFileData,
        vehicle_parameters: VehicleParameters,
        map_factory: AbstractMapFactory,
        period_milliseconds: int = 5000,
        radius: float = 300.0,
    ):
        """
        Scenario simulation tile.
        :param doc: Bokeh HTML document.
        :param experiment_file_data: Experiment file data.
        :param vehicle_parameters: Ego pose parameters.
        :param map_factory: Map factory for building maps.
        :param period_milliseconds: Milli seconds to update the tile.
        :param radius: Map radius.
        """
        self._doc = doc
        self._vehicle_parameters = vehicle_parameters
        self._map_factory = map_factory
        self._experiment_file_data = experiment_file_data
        self._period_milliseconds = period_milliseconds
        self._radius = radius
        self._selected_scenario_keys: List[SimulationScenarioKey] = []
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._maps: Dict[str, AbstractMap] = {}
        self._figures: List[SimulationFigure] = []

    def _create_initial_figure(self, figure_index: int, backend: Optional[str] = "webgl") -> SimulationFigure:
        """
        Create an initial Bokeh figure.
        :param figure_index: Figure index.
        :param backend: Bokeh figure backend.
        :return: A Bokeh figure.
        """
        experiment_path = Path(
            self._experiment_file_data.file_paths[
                self._selected_scenario_keys[figure_index].nuboard_file_index
            ].metric_main_path
        )
        planner_name = self._selected_scenario_keys[figure_index].planner_name
        presented_planner_name = planner_name + f' ({experiment_path.stem})'
        simulation_figure = figure(
            x_range=(-self._radius, self._radius),
            y_range=(-self._radius, self._radius),
            width=simulation_tile_style["figure_sizes"][0],
            height=simulation_tile_style["figure_sizes"][1],
            title=f"{presented_planner_name}",
            tools=["pan", "wheel_zoom", "save", "reset"],
            match_aspect=True,
            active_scroll="wheel_zoom",
            margin=simulation_tile_style["figure_margins"],
            background_fill_color=simulation_tile_style["background_color"],
            output_backend=backend,
        )
        simulation_figure.axis.visible = False
        simulation_figure.xgrid.visible = False
        simulation_figure.ygrid.visible = False
        simulation_figure.title.text_font_size = simulation_tile_style["figure_title_text_font_size"]
        slider = Slider(
            start=0,
            end=1,
            value=0,
            step=1,
            title="Frame",
            margin=simulation_tile_style["slider_margins"],
            css_classes=["scenario-frame-slider"],
        )
        slider.on_change("value", partial(self._slider_on_change, figure_index=figure_index))
        video_button = Button(
            label="Render video",
            margin=simulation_tile_style["video_button_margins"],
            css_classes=["scenario-video-button"],
        )
        video_button.on_click(partial(self._video_button_on_click, figure_index=figure_index))

        simulation_figure_data = SimulationFigure(
            figure=simulation_figure,
            slider=slider,
            video_button=video_button,
            vehicle_parameters=self._vehicle_parameters,
            planner_name=planner_name,
        )

        return simulation_figure_data

    def _map_api(self, map_name: str) -> AbstractMap:
        """
        Get a map api.
        :param map_name: Map name.
        :return Map api.
        """
        if map_name not in self._maps:
            self._maps[map_name] = self._map_factory.build_map_from_name(map_name)

        return self._maps[map_name]

    def init_simulations(self) -> None:
        """Initialization of the visualization of simulation panel."""
        self._figures = []
        for figure_index in range(len(self._selected_scenario_keys)):
            simulation_figure = self._create_initial_figure(figure_index=figure_index)
            self._figures.append(simulation_figure)

    @property
    def figures(self) -> List[SimulationFigure]:
        """
        Access bokeh figures.
        :return A list of bokeh figures.
        """
        return self._figures

    def _render_simulation_layouts(self) -> List[SimulationData]:
        """
        Render simulation layouts.
        :return: A list of columns or rows.
        """
        grid_layouts: List[SimulationData] = []
        for simulation_figure in self.figures:
            grid_layouts.append(
                SimulationData(
                    planner_name=simulation_figure.planner_name,
                    plot=gridplot(
                        [[simulation_figure.slider], [simulation_figure.figure], [simulation_figure.video_button]],
                        toolbar_location="left",
                    ),
                )
            )
        return grid_layouts

    def render_simulation_tiles(self, selected_scenario_keys: List[SimulationScenarioKey]) -> List[SimulationData]:
        """
        Render simulation tiles.
        :param selected_scenario_keys: A list of selected scenario keys.
        :return A list of bokeh layouts.
        """
        self._selected_scenario_keys = selected_scenario_keys
        self.init_simulations()
        self._read_files()

        for main_figure in self._figures:
            self._render_scenario(main_figure)

        layouts = self._render_simulation_layouts()
        return layouts

    def _read_files(self) -> None:
        """Read all simulation files to memory."""
        for figure_index, simulation_scenario_key in enumerate(self._selected_scenario_keys):
            sorted_files = sorted(simulation_scenario_key.files, reverse=False)
            if len(sorted_files) == 0:
                raise RuntimeError("No files were found!")

            # Deduce the type of files
            first_file = sorted_files[0]
            serialization = _extract_serialization_type(first_file)
            main_figure = self._figures[figure_index]

            if len(sorted_files) > 1:
                # Load scenes from all the available files
                for file in sorted_files:
                    main_figure.scenes[file] = _load_data(file, serialization)
            else:
                # Load all scenes in one go
                file = first_file
                scenes = _load_data(file, serialization)
                scenes = scenes if isinstance(scenes, list) else [scenes]
                for scene in scenes:
                    timestamp_us = pathlib.Path(str(scene["ego"]["timestamp_us"]))
                    main_figure.scenes[timestamp_us] = scene

    @gen.coroutine
    @without_document_lock
    def _video_button_on_click(self, figure_index: int) -> None:
        """
        Callback to video button click event.
        Note that this callback in run on a background thread.
        :param figure_index: Figure index.
        """
        self._figures[figure_index].video_button.disabled = True
        self._figures[figure_index].video_button.label = "Rendering video now..."

        self._executor.submit(self._video_button_next_tick, figure_index)

    def _reset_video_button(self, figure_index: int) -> None:
        """
        Reset a video button after exporting is done.
        :param figure_index: Figure index.
        """
        self.figures[figure_index].video_button.label = "Render video"
        self.figures[figure_index].video_button.disabled = False

    def _update_video_button_label(self, figure_index: int, label: str) -> None:
        """
        Update a video button label to show progress when rendering a video.
        :param figure_index: Figure index.
        :param label: New video button text.
        """
        self.figures[figure_index].video_button.label = label

    def _video_button_next_tick(self, figure_index: int) -> None:
        """
        Synchronous callback to the video button on click event.
        :param figure_index: Figure index.
        """
        if not len(self._figures):
            return

        images = []
        scenario_name = self._selected_scenario_keys[figure_index].scenario_name
        scenario_type = self._selected_scenario_keys[figure_index].scenario_type
        planner_name = self._selected_scenario_keys[figure_index].planner_name
        video_name = scenario_type + "_" + planner_name + "_" + scenario_name + ".avi"
        video_path = Path(self._experiment_file_data.file_paths[figure_index].simulation_main_path) / "video_screenshot"
        if not video_path.exists():
            video_path.mkdir(parents=True, exist_ok=True)
        video_save_path = video_path / video_name

        scenes = self.figures[figure_index].scenes
        scene_keys = list(scenes.keys())
        scene_states = scenes[scene_keys[0]]
        database_interval = scene_states.get("database_interval", None)
        selected_simulation_figure = self._figures[figure_index]

        try:
            if len(selected_simulation_figure.ego_state_plot.data_sources):
                chrome_options = webdriver.ChromeOptions()
                chrome_options.headless = True
                driver = webdriver.Chrome(chrome_options=chrome_options)
                shape = None
                simulation_figure = self._create_initial_figure(figure_index=figure_index, backend="canvas")
                # Copy the data sources
                simulation_figure.copy_datasources(selected_simulation_figure)
                self._render_scenario(main_figure=simulation_figure)
                length = len(selected_simulation_figure.ego_state_plot.data_sources)
                for frame_index in tqdm(range(length), desc="Rendering video"):
                    self._render_plots(main_figure=simulation_figure, frame_index=frame_index)
                    image = get_screenshot_as_png(column(simulation_figure.figure), driver=driver)
                    shape = image.size
                    images.append(image)
                    label = f"Rendering video now... ({frame_index}/{length})"
                    self._doc.add_next_tick_callback(
                        partial(self._update_video_button_label, figure_index=figure_index, label=label)
                    )

                fourcc = cv2.VideoWriter_fourcc("M", "P", "E", "G")
                if database_interval:
                    fps = 1 / database_interval
                else:
                    fps = 20  # Assume default fps is 20
                video_obj = cv2.VideoWriter(filename=str(video_save_path), fourcc=fourcc, fps=fps, frameSize=shape)
                for index, image in enumerate(images):
                    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    video_obj.write(cv2_image)

                video_obj.release()
                logger.info("Video saved to %s" % str(video_save_path))
        except (RuntimeError, Exception) as e:
            logger.error("%s" % e)

        self._doc.add_next_tick_callback(partial(self._reset_video_button, figure_index=figure_index))

    def _slider_on_change(self, attr: str, old: int, new: int, figure_index: int) -> None:
        """
        Helper function when a slider changes.
        :param attr: Attribute name.
        :param old: Old value.
        :param new: New value.
        :param figure_index: Figure index.
        """
        if new != len(self._figures[figure_index].scenes):
            self._render_plots(main_figure=self._figures[figure_index], frame_index=new)

    def _render_scenario(self, main_figure: SimulationFigure) -> None:
        """
        Render scenario.
        :param main_figure: Simulation figure object.
        """
        scenes = main_figure.scenes
        files = list(scenes.keys())
        if len(files) == 0:
            return

        # Load the first file only.
        scene_states = scenes[files[0]]
        self._render_map(main_figure=main_figure, scene=scene_states)

        expert_ego_trajectory = None
        if "ego_expert_trajectory" in scene_states["trajectories"]:
            expert_ego_trajectory = scene_states["trajectories"]["ego_expert_trajectory"]

        if expert_ego_trajectory is not None:
            self._render_expert_trajectory(main_figure=main_figure, expert_ego_trajectory=expert_ego_trajectory)

        if scene_states["goal"] is not None:
            main_figure.render_mission_goal(mission_goal_state=scene_states["goal"])

        # Must be updated after drawing maps
        main_figure.update_data_sources()
        self._render_plots(main_figure=main_figure, frame_index=0)

    def _render_map(self, main_figure: SimulationFigure, scene: Dict[str, Any]) -> None:
        """
        Render a map.
        :param main_figure: Simulation figure.
        :param scene: A dictionary of scene info.
        """
        map_name = scene["map_name"]
        map_api = self._map_api(map_name)
        layer_names = [
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        ]
        ego_pose = scene["ego"]["pose"]
        center = Point2D(ego_pose[0], ego_pose[1])

        nearest_vector_map = map_api.get_proximal_map_objects(center, self._radius, layer_names)
        # Filter out stop polygons in turn stop
        if SemanticMapLayer.STOP_LINE in nearest_vector_map:
            stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
            nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
                stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
            ]

        # Draw polygons
        polygon_layer_names = [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.LANE]),
            (SemanticMapLayer.INTERSECTION, simulation_map_layer_color[SemanticMapLayer.INTERSECTION]),
            (SemanticMapLayer.STOP_LINE, simulation_map_layer_color[SemanticMapLayer.STOP_LINE]),
            (SemanticMapLayer.CROSSWALK, simulation_map_layer_color[SemanticMapLayer.CROSSWALK]),
            (SemanticMapLayer.WALKWAYS, simulation_map_layer_color[SemanticMapLayer.WALKWAYS]),
            (SemanticMapLayer.CARPARK_AREA, simulation_map_layer_color[SemanticMapLayer.CARPARK_AREA]),
        ]

        for layer_name, color in polygon_layer_names:
            layer = nearest_vector_map[layer_name]
            map_polygon = MapPoint(point_2d=[])

            for map_obj in layer:
                coords = map_obj.polygon.exterior.coords
                points = [Point2D(x=x, y=y) for x, y in coords]
                map_polygon.point_2d.append(points)

            polygon_source = ColumnDataSource(
                dict(
                    xs=map_polygon.polygon_xs,
                    ys=map_polygon.polygon_ys,
                )
            )
            main_figure.map_polygon_plots[layer_name.name] = main_figure.figure.multi_polygons(
                xs="xs",
                ys="ys",
                fill_color=color["fill_color"],
                fill_alpha=color["fill_color_alpha"],
                line_color=color["line_color"],
                source=polygon_source,
            )

        # Draw lines
        line_layer_names = [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.BASELINE_PATHS]),
            (SemanticMapLayer.LANE_CONNECTOR, simulation_map_layer_color[SemanticMapLayer.LANE_CONNECTOR]),
        ]
        for layer_name, color in line_layer_names:
            layer = nearest_vector_map[layer_name]
            map_line = MapPoint(point_2d=[])
            for map_obj in layer:
                path = map_obj.baseline_path().discrete_path()
                points = [Point2D(x=pose.x, y=pose.y) for pose in path]
                map_line.point_2d.append(points)

            line_source = ColumnDataSource(dict(xs=map_line.line_xs, ys=map_line.line_ys))
            main_figure.map_line_plots[layer_name.name] = main_figure.figure.multi_line(
                xs="xs",
                ys="ys",
                line_color=color["line_color"],
                line_alpha=color["line_color_alpha"],
                line_width=0.5,
                line_dash="dashed",
                source=line_source,
            )

        main_figure.lane_connectors = {
            lane_connector.id: lane_connector for lane_connector in nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]
        }

    @staticmethod
    def _render_expert_trajectory(main_figure: SimulationFigure, expert_ego_trajectory: Dict[str, Any]) -> None:
        """
        Render expert trajectory.
        :param main_figure: Main simulation figure.
        :param expert_ego_trajectory: A list of trajectory states.
        """
        source = extract_source_from_states(expert_ego_trajectory["states"])
        main_figure.render_expert_trajectory(expert_ego_trajectory_state=source)

    def _render_plots(self, main_figure: SimulationFigure, frame_index: int) -> None:
        """
        Render plot with a frame index.
        :param frame_index: A frame index.
        """
        if main_figure.lane_connectors is not None and len(main_figure.lane_connectors):
            main_figure.traffic_light_plot.update_plot(main_figure=main_figure.figure, frame_index=frame_index)

        main_figure.ego_state_plot.update_plot(
            main_figure=main_figure.figure, frame_index=frame_index, radius=self._radius
        )

        # Update ego pose trajectory state.
        main_figure.ego_state_trajectory_plot.update_plot(main_figure=main_figure.figure, frame_index=frame_index)

        # Update agent data sources
        main_figure.agent_state_plot.update_plot(main_figure=main_figure.figure, frame_index=frame_index)

        # Update agent heading data sources
        main_figure.agent_state_heading_plot.update_plot(main_figure=main_figure.figure, frame_index=frame_index)

        main_figure.update_legend()
