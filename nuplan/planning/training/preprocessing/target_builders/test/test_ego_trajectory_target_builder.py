import unittest
from typing import Any, Dict, Union
from unittest.mock import patch

import numpy as np
from hypothesis import example, given
from hypothesis import strategies as st

from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

PATCH_STR = "nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder"

# Defining some upper and lower bounds for testing purposes.
MIN_TIME_INTERVAL = 0.05  # [s]
MAX_TIME_INTERVAL = 1.0  # [s]

MIN_FUTURE_ITERATIONS = 5  # Must be > 2 due to selection of picked_interval_length below.
MAX_FUTURE_ITERATIONS = 100


@st.composite
def _get_valid_test_parameters(draw: Any) -> Dict[str, Union[int, float]]:
    """
    This function implements a strategy to define trajectory / time parameters
    for both MockAbstractScenario and TrajectorySampling, used by EgoTargetTrajectoryBuilder.
    :param draw: The draw function used to select a specific sample given the strategy.
    :return: A dictionary mapping parameters to sampled values to test.
    """
    # Parameter Set for MockAbstractScenario Construction
    test_time_step_samples = st.sampled_from(
        np.arange(MIN_TIME_INTERVAL, MIN_TIME_INTERVAL + MAX_TIME_INTERVAL, MIN_TIME_INTERVAL)
    )

    test_number_of_future_iterations_samples = st.integers(
        min_value=MIN_FUTURE_ITERATIONS,
        max_value=MAX_FUTURE_ITERATIONS,
    )

    # Parameter Set (that can be determined a priori) for TrajectorySampling.
    test_field_to_set_none = st.one_of(st.none(), st.sampled_from(["num_poses", "time_horizon", "interval_length"]))

    # Get a sample of MockAbstractScenario parameters.
    picked_time_step = draw(test_time_step_samples)
    picked_number_of_future_iterations = draw(test_number_of_future_iterations_samples)

    # This is the length of the trajectory in MockAbstractScenario.
    scenario_trajectory_full_horizon = picked_time_step * picked_number_of_future_iterations

    # We make a simplifying assumption that the requested interval_length for TrajectorySampling
    # is a multiple of the MockAbstractScenario time_step.
    picked_interval_length = draw(
        st.sampled_from(
            np.arange(picked_time_step, scenario_trajectory_full_horizon - picked_time_step, picked_time_step)
        )
    )

    # Pick num_poses for TrajectorySampling s.t. we will not exceed the trajectory length in MockAbstractScenario.
    picked_num_poses = draw(
        st.integers(
            min_value=1,
            max_value=int(scenario_trajectory_full_horizon / picked_interval_length),
        )
    )

    # This parameter is not selected but completely defined by the other two parameters.
    picked_trajectory_horizon = picked_interval_length * picked_num_poses

    # This is the full set of all test parameters to use.
    trajectory_sampling_dict = {
        "time_step": picked_time_step,
        "number_of_future_iterations": picked_number_of_future_iterations,
        "num_poses": picked_num_poses,
        "time_horizon": picked_trajectory_horizon,
        "interval_length": picked_interval_length,
    }

    # We may opt to delete a field from TrajectorySampling construction for the case where only 2 params are defined.
    picked_field_to_set_none = draw(test_field_to_set_none)
    if picked_field_to_set_none is not None:
        trajectory_sampling_dict[picked_field_to_set_none] = None

    return trajectory_sampling_dict


class TestEgoTrajectoryTargetBuilder(unittest.TestCase):
    """Test class for EgoTrajectoryTargetBuilder."""

    @given(
        test_parameters=_get_valid_test_parameters(),
    )
    @example(
        test_parameters={
            "time_step": 0.05,
            "number_of_future_iterations": 100,
            "num_poses": 10,
            "time_horizon": 5.0,
            "interval_length": None,
        }
    )
    def test_get_targets(
        self,
        test_parameters: Dict[str, Union[int, float]],
    ) -> None:
        """
        Parametrized test for target trajectory extraction.
        :param test_parameters: The dictionary mapping parameters to sampled values to apply in the test.
        """
        test_scenario = MockAbstractScenario(
            time_step=test_parameters["time_step"],
            number_of_future_iterations=test_parameters["number_of_future_iterations"],
        )

        test_future_trajectory_sampling = TrajectorySampling(
            num_poses=test_parameters["num_poses"],
            time_horizon=test_parameters["time_horizon"],
            interval_length=test_parameters["interval_length"],
        )

        builder = EgoTrajectoryTargetBuilder(future_trajectory_sampling=test_future_trajectory_sampling)

        generated_features = builder.get_targets(test_scenario)

        self.assertEqual(generated_features.num_of_iterations, test_future_trajectory_sampling.num_poses)
        self.assertIsInstance(generated_features, builder.get_feature_type())

    @given(num_poses_diff=st.integers(min_value=-MIN_FUTURE_ITERATIONS + 1, max_value=MIN_FUTURE_ITERATIONS))
    def test_runtime_error_due_to_expected_pose_mismatch(self, num_poses_diff: int) -> None:
        """This tests the edge case if the number of returned poses doesn't match the expected one in the builder."""
        test_scenario = MockAbstractScenario()

        test_future_trajectory_sampling = TrajectorySampling(
            num_poses=max(test_scenario._number_of_future_iterations, MIN_FUTURE_ITERATIONS),
            interval_length=test_scenario._time_step,
        )

        builder = EgoTrajectoryTargetBuilder(future_trajectory_sampling=test_future_trajectory_sampling)

        # We add a return value with an extra pose.  Second dimension corresponds to a state in SE2 serialized.
        with patch(f"{PATCH_STR}.convert_absolute_to_relative_poses") as mock_convert_absolute_to_relative_poses:
            mock_convert_absolute_to_relative_poses.return_value = np.zeros(
                (builder._num_future_poses + num_poses_diff, 3),
                dtype=np.float32,
            )

            if num_poses_diff == 0:
                # Nominal case
                generated_features = builder.get_targets(test_scenario)
                self.assertIsInstance(generated_features, builder.get_feature_type())
            else:
                # We should get a runtime error due to incorrect number of poses returned by the mock.
                with self.assertRaisesRegex(RuntimeError, "Expected.*num poses but got.*"):
                    builder.get_targets(test_scenario)

        # Make sure we used the mock.
        mock_convert_absolute_to_relative_poses.assert_called_once()


if __name__ == "__main__":
    unittest.main()
