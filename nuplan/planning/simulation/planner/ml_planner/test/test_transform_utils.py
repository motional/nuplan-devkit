import unittest
from typing import List
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states


class TestTransformUtils(unittest.TestCase):
    """
    Unit tests for transform_utils.py
    """

    def test_transform_predictions_to_states(self) -> None:
        """
        Test transform predictions to states
        """
        predicted_poses: npt.NDArray[np.float32] = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
        ego_history: List[MagicMock] = []
        for i in range(5):
            s = MagicMock()
            s.time_point.time_s = i * 0.1
            s.car_footprint.vehicle_parameters = VehicleParameters(
                width=2,
                front_length=4,
                rear_length=1,
                cog_position_from_rear_axle=2,
                height=2,
                wheel_base=3,
                vehicle_name='mock',
                vehicle_type='mock',
            )
            s.rear_axle = StateSE2.deserialize([i * 0.1, i * 0.1, np.pi / 4])
            ego_history.append(s)
        future_horizon = 3
        time_interval = 1

        states = transform_predictions_to_states(predicted_poses, ego_history, future_horizon, time_interval)

        # Test the current ego state from history is correctly passed to states
        np.testing.assert_allclose(ego_history[-1].rear_axle.serialize(), states[0].rear_axle.serialize())

        # Test the predicted pose is corrctly converted to states
        gt_poses = [[0.4 + i * np.cos(np.pi / 4), 0.4 + i * np.sin(np.pi / 4), np.pi / 4] for i in range(1, 4)]
        np.testing.assert_allclose(gt_poses, [s.rear_axle.serialize() for s in states[1:]])

        # Test the time is correct
        np.testing.assert_allclose([0.4, 1.4, 2.4, 3.4], [s.time_point.time_s for s in states])

        # Test the velocity
        np.testing.assert_allclose(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [s.dynamic_car_state.center_velocity_2d.array for s in states[1:]],
            rtol=1e-6,
            atol=1e-6,
        )
        # Test the acceleration
        np.testing.assert_allclose(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [s.dynamic_car_state.center_acceleration_2d.array for s in states[1:]],
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == '__main__':
    unittest.main()
