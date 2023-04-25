import tempfile
import unittest
from pathlib import Path

from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import WeightedAverageMetricAggregator
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.metrics.metric_file import MetricFile, MetricFileKey
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback


class TestMetricSummaryCallback(unittest.TestCase):
    """Test metric_summary callback functionality."""

    def set_up_dummy_metric(
        self, metric_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str
    ) -> None:
        """
        Set up dummy metric results.
        :param metric_path: Metric path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        # Set up dummy metric statistics
        statistics = [
            Statistic(
                name="ego_max_acceleration", unit="meters_per_second_squared", value=2.0, type=MetricStatisticsType.MAX
            ),
            Statistic(
                name="ego_min_acceleration", unit="meters_per_second_squared", value=0.0, type=MetricStatisticsType.MIN
            ),
            Statistic(
                name="ego_p90_acceleration", unit="meters_per_second_squared", value=1.0, type=MetricStatisticsType.P90
            ),
        ]
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit="meters_per_second_squared", time_stamps=list(time_stamps), values=list(accel))
        result = MetricStatistics(
            metric_computator="ego_acceleration",
            name="ego_acceleration_statistics",
            statistics=statistics,
            time_series=time_series,
            metric_category="Dynamic",
            metric_score=1,
        )

        # Set up dummy metric file
        key = MetricFileKey(
            metric_name="ego_acceleration",
            scenario_name=scenario_name,
            log_name=log_name,
            scenario_type=scenario_type,
            planner_name=planner_name,
        )

        # Set up a dummy metric engine and save the results to a metric file.
        metric_engine = MetricsEngine(main_save_path=metric_path)

        metric_files = {"ego_acceleration": [MetricFile(key=key, metric_statistics=[result])]}

        metric_engine.write_to_files(metric_files=metric_files)

        # Integrate to a metric file
        metric_file_callback = MetricFileCallback(
            metric_file_output_path=str(metric_path),
            scenario_metric_paths=[str(metric_path)],
            delete_scenario_metric_files=True,
        )
        metric_file_callback.on_run_simulation_end()

    def setUp(self) -> None:
        """Set up a nuboard base tab."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        log_name = 'dummy_log'
        planner_name = "SimplePlanner"
        scenario_type = "Test"
        scenario_name = "Dummy_scene"

        # Set up dummy metric files
        metric_path = Path(self.tmp_dir.name) / 'metrics'
        metric_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_metric(
            metric_path=metric_path,
            log_name=log_name,
            planner_name=planner_name,
            scenario_name=scenario_name,
            scenario_type=scenario_type,
        )
        self.aggregator_save_path = Path(self.tmp_dir.name) / 'aggregator_metric'
        self.weighted_average_metric_aggregator = WeightedAverageMetricAggregator(
            name='weighted_average_metric_aggregator',
            metric_weights={'default': 1.0, 'dummy_metric': 0.5},
            file_name='test_weighted_average_metric_aggregator.parquet',
            aggregator_save_path=self.aggregator_save_path,
            multiple_metrics=[],
        )

        self.metric_statistics_dataframes = {}
        for metric_parquet_file in metric_path.iterdir():
            print(metric_parquet_file)
            data_frame = MetricStatisticsDataFrame.load_parquet(metric_parquet_file)
            self.metric_statistics_dataframes[data_frame.metric_statistic_name] = data_frame

        self.metric_summary_output_path = Path(self.tmp_dir.name) / 'summary'
        self.metric_summary_callback = MetricSummaryCallback(
            metric_save_path=str(metric_path),
            metric_aggregator_save_path=str(self.aggregator_save_path),
            summary_output_path=str(self.metric_summary_output_path),
            pdf_file_name='summary.pdf',
        )

    def test_metric_summary_callback_on_simulation_end(self) -> None:
        """Test on_simulation_end in metric summary callback."""
        # Run metric aggregator
        self.weighted_average_metric_aggregator(metric_dataframes=self.metric_statistics_dataframes)

        # Run summary rendering
        self.metric_summary_callback.on_run_simulation_end()

        pdf_files = self.metric_summary_output_path.rglob("*.pdf")
        # Only one summary pdf
        self.assertEqual(len(list(pdf_files)), 1)

    def tearDown(self) -> None:
        """Remove all temporary folders and files."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
