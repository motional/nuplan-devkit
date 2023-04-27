import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import matplotlib.cm as cmap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

from nuplan.common.utils.io_utils import safe_path_to_string
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
from nuplan.planning.nuboard.tabs.config.histogram_tab_config import (
    HistogramConstantConfig,
    HistogramTabFigureStyleConfig,
    HistogramTabMatPlotLibPlotStyleConfig,
)
from nuplan.planning.nuboard.utils.nuboard_histogram_utils import (
    aggregate_metric_aggregator_dataframe_histogram_data,
    aggregate_metric_statistics_dataframe_histogram_data,
    compute_histogram_edges,
    get_histogram_plot_x_range,
)
from nuplan.planning.nuboard.utils.utils import metric_aggregator_reader, metric_statistics_reader
from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

METRIC_DATAFRAME_TYPE = Dict[str, Union[MetricStatisticsDataFrame, pd.DataFrame]]

logger = logging.getLogger(__name__)


class MetricSummaryCallback(AbstractMainCallback):
    """Callback to render histograms for metrics and metric aggregator."""

    def __init__(
        self,
        metric_save_path: str,
        metric_aggregator_save_path: str,
        summary_output_path: str,
        pdf_file_name: str,
        num_bins: int = 20,
    ):
        """Callback to handle metric files at the end of process."""
        self._metric_save_path = Path(metric_save_path)
        self._metric_aggregator_save_path = Path(metric_aggregator_save_path)
        self._summary_output_path = Path(summary_output_path)

        if not is_s3_path(self._summary_output_path):
            self._summary_output_path.mkdir(parents=True, exist_ok=True)

        self._pdf_file_name = pdf_file_name
        self._num_bins = num_bins

        self._color_index = 0
        color_palette = cmap.get_cmap('Set1').colors + cmap.get_cmap('Set2').colors + cmap.get_cmap('Set3').colors
        self._color_choices = [mcolors.rgb2hex(color) for color in color_palette]

        self._metric_aggregator_dataframes: Dict[str, pd.DataFrame] = {}
        self._metric_statistics_dataframes: Dict[str, MetricStatisticsDataFrame] = {}

    @staticmethod
    def _read_metric_parquet_files(
        metric_save_path: Path, metric_reader: Callable[[Path], Any]
    ) -> METRIC_DATAFRAME_TYPE:
        """
        Read metric parquet files with different readers.
        :param metric_save_path: Metric save path.
        :param metric_reader: Metric reader to read metric parquet files.
        :return A dictionary of {file_index: {file_name: MetricStatisticsDataFrame or pandas dataframe}}.
        """
        metric_dataframes: Dict[str, Union[MetricStatisticsDataFrame, pd.DataFrame]] = defaultdict()
        metric_file = metric_save_path.rglob("*.parquet")
        for file_index, file in enumerate(metric_file):
            try:
                if file.is_dir():
                    continue
                data_frame = metric_reader(file)
                metric_dataframes[file.stem] = data_frame
            except (FileNotFoundError, Exception):
                # Ignore the file
                pass
        return metric_dataframes

    def _aggregate_metric_statistic_histogram_data(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate metric statistic histogram data.
        :return A dictionary of metric names and their aggregated data.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        for dataframe_filename, dataframe in self._metric_statistics_dataframes.items():
            histogram_data_list = aggregate_metric_statistics_dataframe_histogram_data(
                metric_statistics_dataframe=dataframe,
                metric_statistics_dataframe_index=0,
                metric_choices=[],
                scenario_types=None,
            )
            if histogram_data_list:
                data[dataframe.metric_statistic_name] += histogram_data_list

        return data

    def _aggregate_scenario_type_score_histogram_data(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate scenario type score histogram data.
        :return A dictionary of scenario type metric name and their scenario type scores.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        for index, (dataframe_filename, dataframe) in enumerate(self._metric_aggregator_dataframes.items()):
            histogram_data_list = aggregate_metric_aggregator_dataframe_histogram_data(
                metric_aggregator_dataframe=dataframe,
                metric_aggregator_dataframe_index=index,
                scenario_types=['all'],
                dataframe_file_name=dataframe_filename,
            )
            if histogram_data_list:
                data[
                    f'{HistogramConstantConfig.SCENARIO_TYPE_SCORE_HISTOGRAM_NAME}_{dataframe_filename}'
                ] += histogram_data_list

        return data

    def _assign_planner_colors(self) -> Dict[str, Any]:
        """
        Assign colors to planners.
        :return A dictionary of planner and colors.
        """
        planner_color_maps = {}
        for dataframe_filename, dataframe in self._metric_statistics_dataframes.items():
            planner_names = dataframe.planner_names
            for planner_name in planner_names:
                if planner_name not in planner_color_maps:
                    planner_color_maps[planner_name] = self._color_choices[self._color_index % len(self._color_choices)]
                    self._color_index += 1

        return planner_color_maps

    def _save_to_pdf(self, matplotlib_plots: List[Any]) -> None:
        """
        Save a list of matplotlib plots to a pdf file.
        :param matplotlib_plots: A list of matplotlib plots.
        """
        file_name = safe_path_to_string(self._summary_output_path / self._pdf_file_name)
        pp = PdfPages(file_name)
        # Save to pdf
        for fig in matplotlib_plots[::-1]:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close()

    @staticmethod
    def _render_ax_hist(
        ax: Any,
        x_values: npt.NDArray[np.float64],
        x_axis_label: str,
        y_axis_label: str,
        bins: npt.NDArray[np.float64],
        label: str,
        color: str,
        ax_title: str,
    ) -> None:
        """
        Render axis with histogram bins.
        :param ax: Matplotlib axis.
        :param x_values: An array of histogram x-axis values.
        :param x_axis_label: Label in the x-axis.
        :param y_axis_label: Label in the y-axis.
        :param bins: An array of histogram bins.
        :param label: Legend name for the bins.
        :param color: Color for the bins.
        :param ax_title: Axis title.
        """
        ax.hist(x=x_values, bins=bins, label=label, color=color, weights=np.ones(len(x_values)) / len(x_values))
        ax.set_xlabel(x_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.x_axis_label_size)
        ax.set_ylabel(y_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.y_axis_label_size)
        ax.set_title(ax_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.axis_title_size)
        ax.set_ylim(ymin=0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis='both', which='major', labelsize=HistogramTabMatPlotLibPlotStyleConfig.axis_ticker_size)
        ax.legend(fontsize=HistogramTabMatPlotLibPlotStyleConfig.legend_font_size)

    @staticmethod
    def _render_ax_bar_hist(
        ax: Any,
        x_values: Union[npt.NDArray[np.float64], List[str]],
        x_axis_label: str,
        y_axis_label: str,
        x_range: List[str],
        label: str,
        color: str,
        ax_title: str,
    ) -> None:
        """
        Render axis with bar histogram.
        :param ax: Matplotlib axis.
        :param x_values: An array of histogram x-axis values.
        :param x_axis_label: Label in the x-axis.
        :param y_axis_label: Label in the y-axis.
        :param x_range: A list of histogram category names.
        :param label: Legend name for the bins.
        :param color: Color for the bins.
        :param ax_title: Axis title.
        """
        value_categories = {key: 0.0 for key in x_range}
        for value in x_values:
            value_categories[str(value)] += 1.0

        category_names = list(value_categories.keys())
        category_values: List[float] = list(value_categories.values())
        num_scenarios = sum(category_values)
        if num_scenarios != 0:
            category_values = [(value / num_scenarios) * 100 for value in category_values]
            category_values = np.round(category_values, decimals=HistogramTabFigureStyleConfig.decimal_places)
        ax.bar(category_names, category_values, label=label, color=color)
        ax.set_xlabel(x_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.x_axis_label_size)
        ax.set_ylabel(y_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.y_axis_label_size)
        ax.set_title(ax_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.axis_title_size)
        ax.set_ylim(ymin=0)
        ax.tick_params(axis='both', which='major', labelsize=HistogramTabMatPlotLibPlotStyleConfig.axis_ticker_size)
        ax.legend(fontsize=HistogramTabMatPlotLibPlotStyleConfig.legend_font_size)

    def _draw_histogram_plots(
        self,
        planner_color_maps: Dict[str, Any],
        histogram_data_dict: HistogramConstantConfig.HistogramDataType,
        histogram_edges: HistogramConstantConfig.HistogramEdgesDataType,
        n_cols: int = 2,
    ) -> None:
        """
        :param planner_color_maps: Color maps from planner names.
        :param histogram_data_dict: A dictionary of histogram data.
        :param histogram_edges: A dictionary of histogram edges (bins) data.
        :param n_cols: Number of columns in subplot.
        """
        matplotlib_plots = []
        for histogram_title, histogram_data_list in tqdm(histogram_data_dict.items(), desc='Rendering histograms'):
            for histogram_data in histogram_data_list:
                # Get planner color
                color = planner_color_maps.get(histogram_data.planner_name, None)
                if not color:
                    planner_color_maps[histogram_data.planner_name] = self._color_choices[
                        self._color_index % len(self._color_choices)
                    ]
                    color = planner_color_maps.get(histogram_data.planner_name)
                    self._color_index += 1

                n_rows = math.ceil(len(histogram_data.statistics) / n_cols)
                fig_size = min(max(6, len(histogram_data.statistics) // 5 * 5), 24)
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_size, fig_size))
                flatten_axs = axs.flatten()
                fig.suptitle(histogram_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.main_title_size)

                for index, (statistic_name, statistic) in enumerate(histogram_data.statistics.items()):
                    unit = statistic.unit
                    bins: npt.NDArray[np.float64] = np.unique(
                        histogram_edges[histogram_title].get(statistic_name, None)
                    )
                    assert bins is not None, f"Count edge data for {statistic_name} cannot be None!"
                    x_range = get_histogram_plot_x_range(unit=unit, data=bins)
                    values = np.round(statistic.values, HistogramTabFigureStyleConfig.decimal_places)
                    if unit in ["count"]:
                        self._render_ax_bar_hist(
                            ax=flatten_axs[index],
                            x_values=values,
                            x_range=x_range,
                            x_axis_label=unit,
                            y_axis_label='Frequency (%)',
                            label=histogram_data.planner_name,
                            color=color,
                            ax_title=statistic_name,
                        )
                    elif unit in ["bool", "boolean"]:
                        values = ["True" if value else "False" for value in values]
                        self._render_ax_bar_hist(
                            ax=flatten_axs[index],
                            x_values=values,
                            x_range=x_range,
                            x_axis_label=unit,
                            y_axis_label='Frequency (%)',
                            label=histogram_data.planner_name,
                            color=color,
                            ax_title=statistic_name,
                        )
                    else:
                        self._render_ax_hist(
                            ax=flatten_axs[index],
                            x_values=values,
                            bins=bins,
                            x_axis_label=unit,
                            y_axis_label='Frequency (%)',
                            label=histogram_data.planner_name,
                            color=color,
                            ax_title=statistic_name,
                        )

                if n_rows * n_cols != len(histogram_data.statistics.values()):
                    flatten_axs[-1].set_axis_off()
                plt.tight_layout()
                matplotlib_plots.append(fig)

        self._save_to_pdf(matplotlib_plots=matplotlib_plots)

    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        start_time = time.perf_counter()

        # Stop if no metric save path
        if not self._metric_save_path.exists() and not self._metric_aggregator_save_path.exists():
            return

        self._metric_aggregator_dataframes = self._read_metric_parquet_files(
            metric_save_path=self._metric_aggregator_save_path, metric_reader=metric_aggregator_reader
        )
        self._metric_statistics_dataframes = self._read_metric_parquet_files(
            metric_save_path=self._metric_save_path,
            metric_reader=metric_statistics_reader,
        )
        planner_color_maps = self._assign_planner_colors()

        # Aggregate histogram data
        histogram_data_dict = self._aggregate_metric_statistic_histogram_data()
        scenario_type_histogram_data_dict = self._aggregate_scenario_type_score_histogram_data()
        # Integrate them into the same dictionary
        histogram_data_dict.update(scenario_type_histogram_data_dict)

        # Compute edges
        histogram_edge_data = compute_histogram_edges(bins=self._num_bins, aggregated_data=histogram_data_dict)
        self._draw_histogram_plots(
            planner_color_maps=planner_color_maps,
            histogram_data_dict=histogram_data_dict,
            histogram_edges=histogram_edge_data,
        )

        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))
        logger.info('Metric summary: {} [HH:MM:SS]'.format(time_str))
