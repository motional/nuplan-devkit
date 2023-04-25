from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from bokeh.plotting import Figure

from nuplan.planning.nuboard.style import PLOT_PALETTE


@dataclass
class HistogramStatistics:
    """Histogram statistics data."""

    values: npt.NDArray[np.float64]  # An array of values
    unit: str  # Unit
    scenarios: List[str]  # Scenario names


@dataclass
class HistogramData:
    """Histogram data."""

    experiment_index: int  # Experiment index to represent color
    planner_name: str  # Planner name
    statistics: Dict[str, HistogramStatistics]  # Aggregated statistic data
    histogram_file_name: Optional[str] = None  # File name that saves this histogram data


@dataclass
class HistogramFigureData:
    """Histogram figure data."""

    figure_plot: Figure  # Histogram statistic figure
    frequency_array: Optional[npt.NDArray[np.int64]] = None


@dataclass
class HistogramEdgeData:
    """Histogram edge data."""

    unit: str  # Unit
    values: npt.NDArray[np.float64]  # An array of values


@dataclass(frozen=True)
class HistogramConstantConfig:
    """Data class config for constant data in histogram."""

    # Type for histogram aggregated data: {metric name: A list of histogram aggregated data}
    HistogramDataType: ClassVar[Any] = Dict[str, List[HistogramData]]

    # Type for histogram figure data type: {metric name: {metric statistics name: histogram figure data}}
    HistogramFigureDataType: ClassVar[Any] = Dict[str, Dict[str, HistogramFigureData]]

    # Type for histogram edge data type: {metric name: {metric statistic name: histogram figure data}}
    HistogramEdgesDataType: ClassVar[Any] = Dict[str, Dict[str, Optional[npt.NDArray[np.float64]]]]

    # Type for scenario type score histogram
    HistogramScenarioTypeScoreStatisticType: ClassVar[Any] = Dict[str, Dict[str, List[Tuple[float, str]]]]

    PLANNER_CHECKBOX_GROUP_NAME: ClassVar[str] = 'histogram_planner_checkbox_group'
    SCENARIO_TYPE_SCORE_HISTOGRAM_NAME: ClassVar[str] = 'scenario_type_scores'
    HISTOGRAM_TAB_DEFAULT_NUMBER_COLS: ClassVar[int] = 3


@dataclass(frozen=True)
class HistogramTabMatPlotLibPlotStyleConfig:
    """Histogram figure style for matplotlib plot."""

    main_title_size: int = 8
    axis_title_size: int = 6
    y_axis_label_size: int = 5
    x_axis_label_size: int = 5
    legend_font_size: int = 4
    axis_ticker_size: int = 5


@dataclass(frozen=True)
class HistogramTabHistogramBarStyleConfig:
    """Histogram tab bar style configs."""

    line_color: ClassVar[str] = "white"
    fill_alpha: ClassVar[float] = 0.5
    line_alpha: ClassVar[float] = 0.5
    line_width: ClassVar[int] = 3

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {
            'line_color': cls.line_color,
            'fill_alpha': cls.fill_alpha,
            'line_alpha': cls.line_alpha,
            'line_width': cls.line_width,
        }

    @classmethod
    def update_histogram_bar_figure_style(cls, histogram_figure: Figure) -> None:
        """Update histogram figure bar style."""
        histogram_figure.y_range.start = 0
        histogram_figure.legend.background_fill_alpha = 0.3
        histogram_figure.legend.label_text_font_size = "8pt"
        histogram_figure.yaxis.axis_label = "Frequency"
        histogram_figure.grid.grid_line_color = "white"


@dataclass(frozen=True)
class HistogramTabFigureGridPlotStyleConfig:
    """Histogram tab figure grid plot style configs."""

    toolbar_location: ClassVar[str] = "left"

    @classmethod
    def get_config(cls, ncols: int, height: int) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'toolbar_location': cls.toolbar_location, 'ncols': ncols, 'height': height}


@dataclass(frozen=True)
class HistogramTabFigureTitleDivStyleConfig:
    """Histogram tab figure title div style configs."""

    style: ClassVar[Dict[str, str]] = {"font-size": "10pt", "width": "100%", "font-weight": "bold"}

    @classmethod
    def get_config(cls, title: str) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'text': title, 'style': cls.style}


@dataclass(frozen=True)
class HistogramTabFigureStyleConfig:
    """Histogram tab figure style configs."""

    background_fill_color: ClassVar[str] = PLOT_PALETTE["background_white"]
    margin: ClassVar[List[int]] = [10, 20, 20, 30]
    output_backend: ClassVar[str] = 'webgl'
    active_scroll: ClassVar[str] = 'wheel_zoom'
    maximum_plot_width: ClassVar[int] = 1200
    decimal_places: ClassVar[int] = 6

    @classmethod
    def get_config(
        cls, title: str, x_axis_label: str, width: int, height: int, x_range: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {
            'title': title,
            'x_axis_label': x_axis_label,
            'width': width,
            'height': height,
            'x_range': x_range,
            'background_fill_color': cls.background_fill_color,
            'margin': cls.margin,
            'output_backend': cls.output_backend,
            'active_scroll': cls.active_scroll,
        }

    @classmethod
    def update_histogram_figure_style(cls, histogram_figure: Figure) -> None:
        """Update histogram figure style."""
        histogram_figure.title.text_font_size = "8pt"
        histogram_figure.xaxis.axis_label_text_font_size = "8pt"
        histogram_figure.xaxis.major_label_text_font_size = "7pt"
        histogram_figure.yaxis.axis_label_text_font_size = "8pt"
        histogram_figure.yaxis.major_label_text_font_size = "8pt"

        # Rotate the x_axis label with 45 (180/4) degrees
        histogram_figure.xaxis.major_label_orientation = np.pi / 2  # 90 (180 / 2) degrees
        histogram_figure.toolbar.logo = None


@dataclass(frozen=True)
class HistogramTabScenarioTypeMultiChoiceConfig:
    """Config for the histogram tab scenario type multi choice tag."""

    name: ClassVar[str] = 'histogram_scenario_type_multi_choice'
    css_classes: ClassVar[List[str]] = ['scenario-type-multi-choice']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'name': cls.name, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class HistogramTabMetricNameMultiChoiceConfig:
    """Config for the histogram tab metric name multi choice tag."""

    name: ClassVar[str] = 'histogram_metric_name_multi_choice'
    css_classes: ClassVar[List[str]] = ['metric-name-multi-choice']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'name': cls.name, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class HistogramTabBinSpinnerConfig:
    """Config for the histogram tab bin spinner tag."""

    mode: ClassVar[str] = 'int'
    placeholder: ClassVar[str] = 'Number of bins (default: 10, max: 100)'
    low: ClassVar[int] = 1
    high: ClassVar[int] = 100
    name: ClassVar[str] = 'histogram_bin_spinner'
    css_classes: ClassVar[List[str]] = ['histogram-bin-spinner']
    default_bins: ClassVar[int] = 10

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {
            'mode': cls.mode,
            'placeholder': cls.placeholder,
            'low': cls.low,
            'high': cls.high,
            'name': cls.name,
            'css_classes': cls.css_classes,
        }


@dataclass(frozen=True)
class HistogramTabDefaultDivConfig:
    """Config for the histogram tab default div tag."""

    text: ClassVar[str] = '<p> No histogram results, please add more experiments or adjust the search filter.</p>'
    margin: ClassVar[List[int]] = [5, 5, 5, 30]
    css_classes: ClassVar[List[str]] = ['histogram-default-div']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'text': cls.text, 'margin': cls.margin, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class HistogramTabPlotConfig:
    """Config for the histogram tab plot column tag."""

    css_classes: ClassVar[List[str]] = ['histogram-plots']
    name: ClassVar[str] = 'histogram_plots'
    default_width: ClassVar[int] = 800

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'name': cls.name, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class HistogramTabModalQueryButtonConfig:
    """Config for the histogram tab modal query button tag."""

    name: ClassVar[str] = 'histogram_modal_query_btn'
    label: ClassVar[str] = 'Search Results'
    css_classes: ClassVar[List[str]] = ['btn', 'btn-primary', 'modal-btn', 'histogram-tab-modal-query-btn']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'name': cls.name, 'label': cls.label, 'css_classes': cls.css_classes}
