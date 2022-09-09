from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from bokeh.plotting import figure


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


@dataclass
class HistogramFigureData:
    """Histogram figure data."""

    figure_plot: figure  # Histogram statistic figure
    frequency_array: Optional[npt.NDArray[np.int64]] = None


@dataclass
class HistogramEdgeData:
    """Histogram edge data."""

    unit: str  # Unit
    values: npt.NDArray[np.float64]  # An array of values


# Type for histogram aggregated data: {metric name: A list of histogram aggregated data}
HistogramDataType = Dict[str, List[HistogramData]]

# Type for histogram figure data type: {metric name: {metric statistics name: histogram figure data}}
HistogramFigureDataType = Dict[str, Dict[str, HistogramFigureData]]

# Type for histogram edge data type: {metric name: {metric statistic name: histogram figure data}}
HistogramEdgesDataType = Dict[str, Dict[str, Optional[npt.NDArray[np.float64]]]]

PLANNER_CHECKBOX_GROUP_NAME = 'histogram_planner_checkbox_group'


@dataclass(frozen=True)
class HistogramTabScenarioTypeMultiChoiceConfig:
    """Config for the histogram tab scenario type multi choice tag."""

    option_limit: ClassVar[int] = 10
    name: ClassVar[str] = 'histogram_scenario_type_multi_choice'
    css_classes: ClassVar[List[str]] = ['scenario-type-multi-choice']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'option_limit': cls.option_limit, 'name': cls.name, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class HistogramTabMetricNameMultiChoiceConfig:
    """Config for the histogram tab metric name multi choice tag."""

    option_limit: ClassVar[int] = 10
    name: ClassVar[str] = 'histogram_metric_name_multi_choice'
    css_classes: ClassVar[List[str]] = ['metric-name-multi-choice']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'option_limit': cls.option_limit, 'name': cls.name, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class HistogramTabBinSpinnerConfig:
    """Config for the histogram tab bin spinner tag."""

    mode: ClassVar[str] = 'int'
    placeholder: ClassVar[str] = 'Number of bins (default: 20)'
    low: ClassVar[int] = 1
    name: ClassVar[str] = 'histogram_bin_spinner'
    css_classes: ClassVar[List[str]] = ['histogram-bin-spinner']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {
            'mode': cls.mode,
            'placeholder': cls.placeholder,
            'low': cls.low,
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

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'name': cls.name, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class HistogramTabModalQueryButtonConfig:
    """Config for the histogram tab modal query button tag."""

    name: ClassVar[str] = 'histogram_modal_query_btn'
    label: ClassVar[str] = 'Query Scenario'
    css_classes: ClassVar[List[str]] = ['btn', 'btn-primary', 'modal-btn', 'histogram-tab-modal-query-btn']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'name': cls.name, 'label': cls.label, 'css_classes': cls.css_classes}
