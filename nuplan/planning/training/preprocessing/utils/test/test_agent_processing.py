import unittest
from typing import List

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects, TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features,
    compute_yaw_rate_from_states,
    extract_and_pad_agent_poses,
    extract_and_pad_agent_sizes,
    extract_and_pad_agent_velocities,
    filter_agents,
)


def _create_ego_trajectory(num_frames: int) -> List[EgoState]:
    """
    Generate a dummy ego trajectory
    :param num_frames: length of the trajectory to be generate
    """
    return [
        EgoState.build_from_rear_axle(
            StateSE2(step, step, step),
            rear_axle_velocity_2d=StateVector2D(step, step),
            rear_axle_acceleration_2d=StateVector2D(step, step),
            tire_steering_angle=step,
            time_point=TimePoint(step),
            vehicle_parameters=get_pacifica_parameters(),
        )
        for step in range(num_frames)
    ]


def _create_scene_object(token: str, object_type: TrackedObjectType) -> Agent:
    """
    :param token: a unique instance token
    :param object_type: agent type.
    :return: a random Agent
    """
    scene = SceneObject.make_random(token, object_type)
    return Agent(
        tracked_object_type=object_type,
        oriented_box=scene.box,
        velocity=StateVector2D(0, 0),
        metadata=SceneObjectMetadata(token=token, track_token=token, track_id=None, timestamp_us=0),
    )


def _create_tracked_objects(
    num_frames: int, num_agents: int, object_type: TrackedObjectType = TrackedObjectType.VEHICLE
) -> List[TrackedObjects]:
    """
    Generate dummy agent trajectories
    :param num_frames: length of the trajectory to be generate
    :param num_agents: number of agents to generate
    :param object_type: agent type.
    :return: agent trajectories [num_frames, num_agents, 1]
    """
    return [
        TrackedObjects([_create_scene_object(str(num), object_type) for num in range(num_agents)])
        for _ in range(num_frames)
    ]


class TestAgentsFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs features with vectorized agent information."""

    def setUp(self) -> None:
        """Set up test case."""
        self.num_frames = 8
        self.num_agents = 10
        self.num_missing_agents = 2

        self.agent_trajectories = [
            *_create_tracked_objects(5, self.num_agents),
            *_create_tracked_objects(3, self.num_agents - self.num_missing_agents),
        ]
        self.time_stamps = [TimePoint(step) for step in range(self.num_frames)]

    def test_build_ego_features(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        ego_trajectory = _create_ego_trajectory(num_frames)
        ego_features = build_ego_features(ego_trajectory)

        self.assertEqual(ego_features.shape[0], num_frames)
        self.assertEqual(ego_features.shape[1], 3)
        self.assertTrue(np.allclose(ego_features[0], np.array([0, 0, 0])))

        ego_features_reversed = build_ego_features(ego_trajectory, reverse=True)
        self.assertTrue(np.allclose(ego_features_reversed[-1], np.array([0, 0, 0])))

    def test_extract_and_pad_agent_poses(self) -> None:
        """
        Test when there is agent pose trajectory is incomplete
        """
        padded_poses, availability = extract_and_pad_agent_poses(self.agent_trajectories)
        availability = np.asarray(availability)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses])  # type: ignore

        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)  # pose dimension

        # test availability
        self.assertEqual(len(availability.shape), 2)
        self.assertEqual(availability.shape[0], self.num_frames)
        self.assertEqual(availability.shape[1], self.num_agents)
        self.assertTrue((availability[:5, :]).all())  # the first 5 frames should be available for all agents
        # the non-missing agents should be true for all time steps
        self.assertTrue((availability[:, : self.num_agents - self.num_missing_agents]).all())
        # missing agents after 5th frames will be not available
        self.assertTrue((~availability[5:, -self.num_missing_agents :]).all())

        padded_poses_reversed, availability_reversed = extract_and_pad_agent_poses(
            self.agent_trajectories[::-1], reverse=True
        )
        availability_reversed = np.asarray(availability_reversed)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses_reversed])
        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)  # pose dimension

        self.assertEqual(len(availability_reversed.shape), 2)
        self.assertEqual(availability_reversed.shape[0], self.num_frames)
        self.assertEqual(availability_reversed.shape[1], self.num_agents)
        # the last 5 frames should be available for all agents
        self.assertTrue((availability_reversed[-5:, :]).all())
        # the non-missing agents should be true for all time steps
        self.assertTrue((availability_reversed[:, : self.num_agents - self.num_missing_agents]).all())
        # missing agents in the first 3 frames will be not available
        self.assertTrue((~availability_reversed[:3, -self.num_missing_agents :]).all())

    def test_extract_and_pad_agent_sizes(self) -> None:
        """
        Test when there is agent size trajectory is incomplete
        """
        padded_sizes, _ = extract_and_pad_agent_sizes(self.agent_trajectories)
        stacked_sizes = np.stack(padded_sizes)  # type: ignore

        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)

        padded_sizes_reversed, _ = extract_and_pad_agent_sizes(self.agent_trajectories[::-1], reverse=True)
        stacked_sizes = np.stack(padded_sizes_reversed)

        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)

    def test_extract_and_pad_agent_velocities(self) -> None:
        """
        Test when there is agent velocity trajectory is incomplete
        """
        padded_velocities, _ = extract_and_pad_agent_velocities(self.agent_trajectories)
        stacked_velocities = np.stack(
            [[agent.serialize() for agent in frame] for frame in padded_velocities]
        )  # type: ignore

        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)

        padded_velocities_reversed, _ = extract_and_pad_agent_velocities(self.agent_trajectories[::-1], reverse=True)
        stacked_velocities = np.stack([[agent.serialize() for agent in frame] for frame in padded_velocities_reversed])

        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)

    def test_compute_yaw_rate_from_states(self) -> None:
        """
        Test computing yaw from the agent pose trajectory
        """
        padded_poses, _ = extract_and_pad_agent_poses(self.agent_trajectories)
        yaw_rates = compute_yaw_rate_from_states(padded_poses, self.time_stamps)

        self.assertEqual(yaw_rates.transpose().shape[0], self.num_frames)
        self.assertEqual(yaw_rates.transpose().shape[1], self.num_agents)

    def test_filter_agents(self) -> None:
        """
        Test agent filtering
        """
        num_frames = 8
        num_agents = 5
        missing_agents = 2

        tracked_objects_history = [
            *_create_tracked_objects(num_frames=5, num_agents=num_agents, object_type=TrackedObjectType.VEHICLE),
            *_create_tracked_objects(
                num_frames=2, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.BICYCLE
            ),
            *_create_tracked_objects(
                num_frames=1, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.VEHICLE
            ),
        ]
        filtered_agents = filter_agents(tracked_objects_history)

        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[0].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)

        filtered_agents = filter_agents(tracked_objects_history, reverse=True)

        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[-1].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)


if __name__ == '__main__':
    unittest.main()
