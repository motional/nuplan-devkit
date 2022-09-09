import unittest

import numpy as np

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.test.test_utils import get_sample_ego_state
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.submission.proto_converters import (
    interp_traj_from_proto_traj,
    proto_tl_status_data_from_tl_status_data,
    proto_tl_status_type_from_tl_status_type,
    proto_traj_from_inter_traj,
    tl_status_data_from_proto_tl_status_data,
    tl_status_type_from_proto_tl_status_type,
)


class TestProtoConverters(unittest.TestCase):
    """Tests proto converters by checking if composition is idempotent."""

    def test_trajectory_conversions(self) -> None:
        """Tests conversions between trajectory object and messages."""
        trajectory = InterpolatedTrajectory(
            [get_sample_ego_state(StateSE2(0, 1, 2)), get_sample_ego_state(StateSE2(1, 2, 3), time_us=1)]
        )
        result = interp_traj_from_proto_traj(proto_traj_from_inter_traj(trajectory))

        for result_state, trajectory_state in zip(result.get_sampled_trajectory(), trajectory.get_sampled_trajectory()):
            np.allclose(result_state.to_split_state().linear_states, trajectory_state.to_split_state().linear_states)
            np.allclose(result_state.to_split_state().angular_states, trajectory_state.to_split_state().angular_states)

    def test_tl_status_type_conversions(self) -> None:
        """Tests conversions between TL status data and messages."""
        tl_status_type = TrafficLightStatusType.RED
        result = tl_status_type_from_proto_tl_status_type(proto_tl_status_type_from_tl_status_type(tl_status_type))
        self.assertEqual(tl_status_type, result)

    def test_tl_status_data_conversions(self) -> None:
        """Tests conversions between TL status type and messages."""
        tl_status = TrafficLightStatusData(TrafficLightStatusType.RED, 123, 456)
        result = tl_status_data_from_proto_tl_status_data(proto_tl_status_data_from_tl_status_data(tl_status))
        self.assertEqual(tl_status, result)


if __name__ == '__main__':
    unittest.main()
