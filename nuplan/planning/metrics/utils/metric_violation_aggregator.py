from typing import List

import numpy as np
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, MetricViolation, Statistic


def aggregate_metric_violations(metric_violations: List[MetricViolation], metric_computator_name: str,
                                metric_category: str,
                                statistics_name: str) -> MetricStatistics:
    """
    Aggregates (possibly) multiple MetricViolations to a MetricStatistics. All the violations must be of the same metric

    :param metric_violations: The list of violations for a single metric name.
    :param metric_computator_name: name of the extractor
    :param metric_category: category of the metric
    :param statistics_name: name of the statistic
    :return: Statistics about the violations.
    """
    if not metric_violations:
        statistics = {MetricStatisticsType.COUNT: Statistic(name=f"number_of_{statistics_name}", unit="count", value=0)}

    else:
        sample_violation = metric_violations[0]
        name = sample_violation.name
        unit = sample_violation.unit

        extrema = []
        mean_values = []
        durations = []
        for violation in metric_violations:
            # We don't want to aggregate different metrics
            assert name == violation.name
            extrema.append(violation.extremum)
            mean_values.append(violation.mean)
            durations.append(violation.duration)

        max_val = max(extrema)
        # Violations which are instantaneous will be reported with unitary duration, while violations which span over
        # time have the respective duration. If a violation is detected at the last time step will have duration 0
        # and won't be taken into account for the mean.
        mean_val = np.sum([mean_value * duration for mean_value, duration in zip(mean_values, durations)]) / sum(
            durations)

        statistics = {
            MetricStatisticsType.MAX: Statistic(name=f"max_violating_{statistics_name}", unit=unit, value=max_val),
            MetricStatisticsType.MEAN: Statistic(name=f"mean_{statistics_name}", unit=unit, value=mean_val),
            MetricStatisticsType.COUNT: Statistic(name=f"number_of_{statistics_name}", unit="count",
                                                  value=len(metric_violations))}

    return MetricStatistics(metric_computator=metric_computator_name,
                            name=statistics_name,
                            metric_category=metric_category,
                            statistics=statistics,
                            time_series=None)
