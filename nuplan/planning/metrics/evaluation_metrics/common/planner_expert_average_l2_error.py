from typing import List, Optional

import numpy as np

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, Statistic, TimeSeries
from nuplan.planning.metrics.utils.expert_comparisons import compute_traj_errors, compute_traj_heading_errors
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center_with_heading, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class PlannerExpertAverageL2ErrorStatistics(MetricBase):
    """Average displacement error metric between the planned ego pose and expert."""

    def __init__(self, name: str, category: str, comparison_horizon: float, comparison_frequency: float) -> None:
        """
        Initialize the PlannerExpertL2ErrorStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param comparison_horizon: Horizon time in future (s) to find displacement errors.
        :param comparison_frequency: Frequency to sample expert and planner trajectory
        """
        super().__init__(name=name, category=category)
        self._comparison_horizon = comparison_horizon
        self._comparison_frequency = comparison_frequency
        # Store the errors to re-use in high level metrics
        self.maximum_displacement_errors: List[float] = []
        self.final_displacement_errors: List[float] = []
        self.ego_timestamps_sampled: List[int] = []
        self.average_heading_errors: List[float] = []
        self.final_heading_errors: List[float] = []

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[-1].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        # In challenge 1 (Open loop) the expert trajectory is the same as ego's driven trajectory
        ego_states = history.extract_ego_state
        ego_timestamps = extract_ego_time_point(ego_states)

        # Find the frequency at which expert trajectory is sampled and the step size for sampling both the expert and planner trajectory
        s_to_us_factor = 1e6
        expert_frequency = s_to_us_factor / np.mean(np.diff(ego_timestamps))
        step_size = int(expert_frequency / self._comparison_frequency)
        ego_traj_poses = extract_ego_center_with_heading(ego_states[0::step_size])

        # Extract planner proposed trajectory at each sampled frame up to self._comparison_horizon seconds before the end of scenario
        curated_timepoint_index = np.sum(
            ego_timestamps + self._comparison_horizon * s_to_us_factor <= ego_timestamps[-1]
        )
        sampled_indices = np.arange(0, curated_timepoint_index, step_size)
        planned_trajectories = (history.data[index].trajectory for index in sampled_indices)
        self.ego_timestamps_sampled = ego_timestamps[0::step_size]

        # Find displacement error between the proposed planner trajectory and expert driven trajectory for all sampled frames during the scenario
        # in which expert trajectory is available up to self._comparison_horizon seconds in the future
        average_displacement_errors = []
        maximum_displacement_errors = []
        final_displacement_errors = []
        average_heading_errors = []
        final_heading_errors = []
        for curr_frame, curr_ego_planned_traj in enumerate(planned_trajectories):
            # Interpolate planner proposed trajectory at the same timepoints where expert states are available
            planner_interpolated_traj = list(
                curr_ego_planned_traj.get_state_at_time(TimePoint(timestamp))
                for timestamp in self.ego_timestamps_sampled[curr_frame:]
                if timestamp <= self._comparison_horizon * s_to_us_factor + curr_ego_planned_traj.start_time.time_us
            )
            planner_interpolated_traj_poses = extract_ego_center_with_heading(planner_interpolated_traj)
            # Find displacement errors between the proposed planner trajectory and expert driven trajectory for the current frame up to self._comparison_horizon seconds in the future
            displacement_errors = compute_traj_errors(
                ego_traj=planner_interpolated_traj_poses,
                expert_traj=ego_traj_poses[curr_frame : curr_frame + len(planner_interpolated_traj) + 1],
                heading_diff_weight=0,
            )
            heading_errors = compute_traj_heading_errors(
                ego_traj=planner_interpolated_traj_poses,
                expert_traj=ego_traj_poses[curr_frame : curr_frame + len(planner_interpolated_traj) + 1],
            )
            average_displacement_errors.append(np.mean(displacement_errors))
            maximum_displacement_errors.append(np.max(displacement_errors))
            final_displacement_errors.append(displacement_errors[-1])
            average_heading_errors.append(np.mean(heading_errors))
            final_heading_errors.append(heading_errors[-1])

        time_series = TimeSeries(
            unit='meters',
            time_stamps=self.ego_timestamps_sampled[: len(average_displacement_errors)],
            values=average_displacement_errors,
        )

        metric_statistics = self._compute_time_series_statistic(time_series=time_series)

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=metric_statistics, scenario=scenario, time_series=time_series
        )
        # Save to re-use in high level metrics
        self.maximum_displacement_errors = maximum_displacement_errors
        self.final_displacement_errors = final_displacement_errors
        self.average_heading_errors = average_heading_errors
        self.final_heading_errors = final_heading_errors

        return results
