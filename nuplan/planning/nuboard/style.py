from typing import Any, Dict

from nuplan.common.maps.abstract_map import SemanticMapLayer

MOTIONAL_PALETTE: Dict[str, str] = {
    'motional_purple': '#5C48F6',
    'hyundai_blue': '#0A2972',
    'navy': '#030A14',
    'medium_asphalt': '#828791',
    'light_asphalt': '#D2DCDC',
    'dark_asphalt': '#40444A',
    'solid_aqua': '#00f6ff',
}

PLOT_PALETTE: Dict[str, str] = {
    'font_lavender_gray': '#adb9e3',
    'font_white': '#c9eaf1',
    'background_white': '#fafafa',
    'background_black': '#000000',
    'chart_green': '#00FF73',
    'chart_yellow': '#E2FF1A',
}

base_tab_style: Dict[str, Any] = {
    'search_criteria_title_width': 300,
    'search_criteria_margin': [20, 0, 0, 20],
    'fig_frame_minimum_height': 800,
    'search_criteria_height': 600,
    'plot_sizes': (300, 400),
    'plot_frame_sizes': (1200, 1200)
}

simulation_tile_style: Dict[str, Any] = {
    'figure_sizes': (800, 800),
    'figure_margins': [10, 20, 20, 50],
    'figure_title_text_font_size': "14pt",
    'slider_margins': [10, 20, 20, 50],
    'background_color': '#000000',
    'mission_goal_color': 'red',
    'expert_color': 'red',
    'agent_color': 'green',
    'planner_color': 'white',
    'mission_goal_size': 20,
    'mission_goal_alpha': 0.2,
    'mission_goal_line_width': 2,
    'expert_trajectory_line_width': 4,
    'planner_line_width': 2,
    'agent_line_width': 1,
    'agent_trajectory_color': 'green',
    'agent_trajectory_line_width': 2,
}

simulation_map_layer_color: Dict[SemanticMapLayer, Any] = {
    SemanticMapLayer.LANE: {'fill_color': '#26334c',
                            'fill_color_alpha': 0.5,
                            'line_color': '#2d3ea7'},
    SemanticMapLayer.INTERSECTION: {'fill_color': '#3e4348',
                                    'fill_color_alpha': 0.5,
                                    'line_color': '#7c8691'},
    SemanticMapLayer.STOP_LINE: {'fill_color': '#7f0000',
                                 'fill_color_alpha': 0.5,
                                 'line_color': '#7f0000'},
    SemanticMapLayer.CROSSWALK: {'fill_color': '#121212',
                                 'fill_color_alpha': 1.0,
                                 'line_color': '#bdbdbd'},
    SemanticMapLayer.BASELINE_PATHS: {'line_color': 'lightblue',
                                      'line_color_alpha': 0.5},
    SemanticMapLayer.LANE_CONNECTOR: {'line_color': 'lightblue',
                                      'line_color_alpha': 0.5}
}

configuration_tab_style: Dict[str, Any] = {
    'folder_path_input_margin': [15, 0, 50, 30],
    'folder_path_selection_margin': [15, 0, 0, 30],
    'main_board_layout_height': 600
}

histogram_tab_style: Dict[str, Any] = {
    'statistic_figure_margin': [10, 20, 20, 50],
    'statistic_figure_title_text_font_size': "14pt",
    'statistic_figure_xaxis_axis_label_text_font_size': "10pt",
    'statistic_figure_xaxis_major_label_text_font_size': "10pt",
    'statistic_figure_yaxis_axis_label_text_font_size': "10pt",
    'statistic_figure_yaxis_major_label_text_font_size': "10pt",
    'histogram_title_div_margin': [30, 0, 10, 30],
    'quad_line_color': 'white',
    'quad_alpha': 0.5,
    'quad_line_width': 3,
    'pdf_alpha': 0.7,
    'pdf_line_width': 4,
    'cdf_alpha': 0.7,
    'cdf_line_width': 2,
    'plot_legend_background_fill_alpha': 0.3,
    'plot_legend_label_text_font_size': "10pt",
    'plot_yaxis_axis_label': "Frequency",
    'plot_grid_line_color': "white"
}

overview_tab_style: Dict[str, Any] = {
    'table_margins': [20, 0, 0, 50],
    'table_width': 800,
    'table_height': 800
}

scenario_tab_style: Dict[str, Any] = {
    'time_series_figure_margins': [10, 20, 20, 50],
    'time_series_figure_title_text_font_size': "14pt",
    'time_series_figure_xaxis_axis_label_text_font_size': "10pt",
    'time_series_figure_xaxis_major_label_text_font_size': "10pt",
    'time_series_figure_yaxis_axis_label_text_font_size': "10pt",
    'time_series_figure_yaxis_major_label_text_font_size': "10pt",
    'time_series_figure_xaxis_axis_label': "frame",
    'plot_legend_label_text_font_size': "10pt"
}
