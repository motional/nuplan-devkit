import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas

from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import WeightedAverageMetricAggregator
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame


class TestWeightedAverageMetricAggregator(unittest.TestCase):
    """Run weighted average metric aggregator unit tests."""

    def setUp(self) -> None:
        """Set up dummy data and folders."""
        # Create some dummy metric dataframes
        self.metric_scores = [[1, 0.5, 0.8], [0.1, 0.2]]
        dummy_dataframes = [
            pandas.DataFrame(
                {
                    'scenario_name': ['test_1', 'test_2', 'test_3'],
                    'log_name': ['dummy', 'dummy', 'dummy_2'],
                    'scenario_type': ['unknown', 'ego_stop_at_stop_line', 'unknown'],
                    'planner_name': ['simple_planner', 'dummy_planner', 'dummy_planner'],
                    'metric_score': self.metric_scores[0],
                    'metric_score_unit': 'float',
                }
            ),
            pandas.DataFrame(
                {
                    'scenario_name': ['test_1', 'test_3'],
                    'log_name': ['dummy', 'dummy_3'],
                    'scenario_type': ['unknown', 'unknown'],
                    'planner_name': ['simple_planner', 'dummy_planner'],
                    'metric_score': self.metric_scores[1],
                    'metric_score_unit': 'float',
                }
            ),
        ]
        metric_statistic_names = ['dummy_metric', 'second_dummy_metric']
        self.metric_statistic_dataframes = []
        for dummy_dataframe, metric_statistic_name in zip(dummy_dataframes, metric_statistic_names):
            self.metric_statistic_dataframes.append(
                MetricStatisticsDataFrame(
                    metric_statistic_name=metric_statistic_name, metric_statistics_dataframe=dummy_dataframe
                )
            )

        self.tmpdir = tempfile.TemporaryDirectory()
        self.weighted_average_metric_aggregator = WeightedAverageMetricAggregator(
            name='weighted_average_metric_aggregator',
            metric_weights={'default': 1.0, 'dummy_metric': 0.5},
            file_name='test_weighted_average_metric_aggregator.parquet',
            aggregator_save_path=Path(self.tmpdir.name),
            multiple_metrics=[],
        )

    def tearDown(self) -> None:
        """Clean up when unittests end."""
        self.tmpdir.cleanup()

    def test_name(self) -> None:
        """Test if name is expected."""
        self.assertEqual('weighted_average_metric_aggregator', self.weighted_average_metric_aggregator.name)

    def test_final_metric_score(self) -> None:
        """Test if final metric score is expected."""
        # Should return None since didn't run aggregation
        self.assertEqual(None, self.weighted_average_metric_aggregator.final_metric_score)

    def test_aggregated_metric_dataframe(self) -> None:
        """Test if aggregated metric dataframe is expected."""
        # Should return None since didn't run aggregation
        self.assertEqual(None, self.weighted_average_metric_aggregator.aggregated_metric_dataframe)

    def test_aggregation(self) -> None:
        """Test running the aggregation."""
        metric_dataframes = {
            metric_statistic_dataframe.metric_statistic_name: metric_statistic_dataframe
            for metric_statistic_dataframe in self.metric_statistic_dataframes
        }

        self.weighted_average_metric_aggregator(metric_dataframes=metric_dataframes)

        # Test if file exists
        parquet_file = Path(self.tmpdir.name) / 'test_weighted_average_metric_aggregator.parquet'
        self.assertTrue(parquet_file.exists())

        # Load the parquet file
        self.weighted_average_metric_aggregator.read_parquet()

        # Test if aggregated metric dataframe not empty
        aggregated_metric_dataframe = self.weighted_average_metric_aggregator.aggregated_metric_dataframe
        self.assertIsNot(aggregated_metric_dataframe, None)
        self.assertTrue(len(aggregated_metric_dataframe))

        self.assertTrue(np.isnan(aggregated_metric_dataframe['second_dummy_metric'][0]))
        expected_planners = ['dummy_planner', 'simple_planner']
        self.assertEqual(expected_planners, sorted(aggregated_metric_dataframe['planner_name'].unique(), reverse=False))
        self.assertEqual(['weighted_average'], list(aggregated_metric_dataframe['aggregator_type'].unique()))

        expected_values = {
            'dummy_planner': {
                'dummy_metric': [0.5, 0.80, 0.50, 0.80, 0.65],
                'second_dummy_metric': [-1.0, 0.2, -1.0, 0.2, 0.1],
                'score': [0.50, 0.40, 0.50, 0.40, 0.45],
            },
            'simple_planner': {
                'dummy_metric': [1.0, 1.0, 1.0],
                'second_dummy_metric': [0.1, 0.1, 0.1],
                'score': [0.40, 0.40, 0.40],
            },
        }
        for planner in expected_planners:
            planner_metric = aggregated_metric_dataframe[aggregated_metric_dataframe['planner_name'].isin([planner])]
            for name, expected_value in expected_values[planner].items():
                # Fill NaN by -1.0
                planner_values = np.round(planner_metric[name].fillna(-1.0).to_numpy(), 2).tolist()
                self.assertEqual(expected_value, planner_values)

    def test_parquet(self) -> None:
        """Test property."""
        self.assertEqual(
            self.weighted_average_metric_aggregator.parquet_file, self.weighted_average_metric_aggregator._parquet_file
        )


if __name__ == '__main__':
    unittest.main()
