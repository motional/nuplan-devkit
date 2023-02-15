import itertools
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, Statistic, TimeSeries
from nuplan.planning.metrics.utils.expert_comparisons import compute_traj_errors, compute_traj_heading_errors
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center_with_heading, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class PlannerExpertAverageL2ErrorStatistics(MetricBase):
    """Average displacement error metric between the planned ego pose and expert."""

    def __init__(
        self,
        name: str,
        category: str,
        comparison_horizon: List[int],
        comparison_frequency: int,
        max_average_l2_error_threshold: float,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initialize the PlannerExpertL2ErrorStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param comparison_horizon: List of horizon times in future (s) to find displacement errors.
        :param comparison_frequency: Frequency to sample expert and planner trajectory.
        :param max_average_l2_error_threshold: Maximum acceptable error threshold.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self.comparison_horizon = comparison_horizon
        self._comparison_frequency = comparison_frequency
        self._max_average_l2_error_threshold = max_average_l2_error_threshold
        # Store the errors to re-use in high level metrics
        self.maximum_displacement_errors: npt.NDArray[np.float64] = np.array([0])
        self.final_displacement_errors: npt.NDArray[np.float64] = np.array([0])
        self.expert_timestamps_sampled: List[int] = []
        self.average_heading_errors: npt.NDArray[np.float64] = np.array([0])
        self.final_heading_errors: npt.NDArray[np.float64] = np.array([0])
        self.selected_frames: List[int] = [0]

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(max(0, 1 - metric_statistics[-1].value / self._max_average_l2_error_threshold))

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        # Find the frequency at which expert trajectory is sampled and the step size for down-sampling it
        expert_frequency = 1 / scenario.database_interval
        step_size = int(expert_frequency / self._comparison_frequency)
        sampled_indices = list(range(0, len(history.data), step_size))
        # Sample the expert trajectory up to the maximum future horizon needed to compute errors
        expert_states = list(
            itertools.chain(
                list(scenario.get_expert_ego_trajectory())[0::step_size],
                scenario.get_ego_future_trajectory(
                    sampled_indices[-1],
                    max(self.comparison_horizon),
                    max(self.comparison_horizon) // self._comparison_frequency,
                ),
            )
        )
        expert_traj_poses = extract_ego_center_with_heading(expert_states)
        expert_timestamps_sampled = extract_ego_time_point(expert_states)

        # Extract planner proposed trajectory at each sampled frame
        planned_trajectories = list(history.data[index].trajectory for index in sampled_indices)

        # Find displacement error between the proposed planner trajectory and expert driven trajectory for all sampled frames during the scenario
        average_displacement_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        maximum_displacement_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        final_displacement_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        average_heading_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        final_heading_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        for curr_frame, curr_ego_planned_traj in enumerate(planned_trajectories):
            future_horizon_frame = int(curr_frame + max(self.comparison_horizon))
            # Interpolate planner proposed trajectory at the same timepoints where expert states are available
            planner_interpolated_traj = list(
                curr_ego_planned_traj.get_state_at_time(TimePoint(int(timestamp)))
                for timestamp in expert_timestamps_sampled[curr_frame : future_horizon_frame + 1]
                if timestamp <= curr_ego_planned_traj.end_time.time_us
            )
            if len(planner_interpolated_traj) < max(self.comparison_horizon) + 1:
                planner_interpolated_traj = list(
                    itertools.chain(planner_interpolated_traj, [curr_ego_planned_traj.get_sampled_trajectory()[-1]])
                )
                # If planner duration is slightly less than the required horizon due to down-sampling, find expert states at the final timepoint of the planner trajectory for the comparison.
                expert_traj = expert_traj_poses[curr_frame + 1 : future_horizon_frame] + [
                    InterpolatedTrajectory(expert_states).get_state_at_time(curr_ego_planned_traj.end_time).center
                ]
            else:
                expert_traj = expert_traj_poses[curr_frame + 1 : future_horizon_frame + 1]

            planner_interpolated_traj_poses = extract_ego_center_with_heading(planner_interpolated_traj)

            # Find displacement errors between the proposed planner trajectory and expert driven trajectory for the current frame up to maximum comparison_horizon seconds in the future
            displacement_errors = compute_traj_errors(
                planner_interpolated_traj_poses[1:],
                expert_traj,
                heading_diff_weight=0,
            )
            heading_errors = compute_traj_heading_errors(
                planner_interpolated_traj_poses[1:],
                expert_traj,
            )

            for ind, horizon in enumerate(self.comparison_horizon):
                horizon_index = horizon // self._comparison_frequency
                average_displacement_errors[ind, curr_frame] = np.mean(displacement_errors[:horizon_index])
                maximum_displacement_errors[ind, curr_frame] = np.max(displacement_errors[:horizon_index])
                final_displacement_errors[ind, curr_frame] = displacement_errors[horizon_index - 1]
                average_heading_errors[ind, curr_frame] = np.mean(heading_errors[:horizon_index])
                final_heading_errors[ind, curr_frame] = heading_errors[horizon_index - 1]

        # Save to re-use in other metrics
        self.ego_timestamps_sampled = expert_timestamps_sampled[: len(sampled_indices)]
        self.selected_frames = sampled_indices

        results: List[MetricStatistics] = self._construct_open_loop_metric_results(
            scenario,
            self.comparison_horizon,
            self._max_average_l2_error_threshold,
            metric_values=average_displacement_errors,
            name='planner_expert_ADE',
            unit='meter',
            timestamps_sampled=self.ego_timestamps_sampled,
            metric_score_unit=self.metric_score_unit,
            selected_frames=sampled_indices,
        )
        # Save to re-use in high level metrics
        self.maximum_displacement_errors = maximum_displacement_errors
        self.final_displacement_errors = final_displacement_errors
        self.average_heading_errors = average_heading_errors
        self.final_heading_errors = final_heading_errors

        return results
