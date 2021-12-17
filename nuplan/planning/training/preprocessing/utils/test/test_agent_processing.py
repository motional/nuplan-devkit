import unittest
from typing import List

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import build_ego_features, \
    compute_yaw_rate_from_states, extract_and_pad_agent_poses, extract_and_pad_agent_sizes, \
    extract_and_pad_agent_velocities, filter_agents


def _create_ego_trajectory(num_frames: int) -> List[EgoState]:
    """
    Generate a dummy ego trajectory
    :param num_frames: length of the trajectory to be generate
    """
    return [EgoState.from_raw_params(StateSE2(step, step, step),
                                     velocity_2d=StateVector2D(step, step),
                                     acceleration_2d=StateVector2D(step, step),
                                     tire_steering_angle=step,
                                     time_point=TimePoint(step)) for step in range(num_frames)]


def _create_box(token: str, label: int) -> Box3D:
    """
    :param token: a unique instance token
    :param label: agent type label. Typically, 1 == vehicles
    :return: a random Box3D
    """
    box = Box3D.make_random()
    box.token = token
    box.label = label
    return box


def _create_agents(num_frames: int, num_agents: int, label: int = 1) -> List[List[Box3D]]:
    """
    Generate dummy agent trajectories
    :param num_frames: length of the trajectory to be generate
    :param num_agents: number of agents to generate
    :param label: agent type label. Typically, 1 == vehicles
    :return: agent trajectories [num_frames, num_agents, 1]
    """
    return [[_create_box(str(num), label) for num in range(num_agents)] for _ in range(num_frames)]


class TestAgentsFeatureBuilder(unittest.TestCase):

    def setUp(self) -> None:
        self.num_frames = 8
        self.num_agents = 10
        self.num_missing_agents = 2
        self.agent_trajectories = _create_agents(5, self.num_agents) + \
            _create_agents(3, self.num_agents - self.num_missing_agents)
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
        padded_poses = extract_and_pad_agent_poses(self.agent_trajectories)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses])

        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)  # pose dimension

        padded_poses_reversed = extract_and_pad_agent_poses(self.agent_trajectories[::-1], reverse=True)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses_reversed])
        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)  # pose dimension

    def test_extract_and_pad_agent_sizes(self) -> None:
        """
        Test when there is agent size trajectory is incomplete
        """
        padded_sizes = extract_and_pad_agent_sizes(self.agent_trajectories)
        stacked_sizes = np.stack(padded_sizes)

        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)

        padded_sizes_reversed = extract_and_pad_agent_sizes(self.agent_trajectories[::-1], reverse=True)
        stacked_sizes = np.stack(padded_sizes_reversed)

        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)

    def test_extract_and_pad_agent_velocities(self) -> None:
        """
        Test when there is agent velocity trajectory is incomplete
        """
        padded_velocities = extract_and_pad_agent_velocities(self.agent_trajectories)
        stacked_velocities = np.stack([[agent.serialize() for agent in frame] for frame in padded_velocities])

        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)

        padded_velocities_reversed = extract_and_pad_agent_velocities(self.agent_trajectories[::-1], reverse=True)
        stacked_velocities = np.stack([[agent.serialize() for agent in frame] for frame in padded_velocities_reversed])

        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)

    def test_compute_yaw_rate_from_states(self) -> None:
        """
        Test computing yaw from the agent pose trajectory
        """
        padded_poses = extract_and_pad_agent_poses(self.agent_trajectories)
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
        agent_trajectories = _create_agents(num_frames=5, num_agents=num_agents, label=1) + \
            _create_agents(num_frames=2, num_agents=num_agents - missing_agents, label=2) + \
            _create_agents(num_frames=1, num_agents=num_agents - missing_agents, label=1)
        filtered_agents = filter_agents(agent_trajectories)

        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0]), len(agent_trajectories[0]))
        self.assertEqual(len(filtered_agents[5]), num_agents - missing_agents)
        self.assertEqual(len(filtered_agents[-1]), num_agents - missing_agents)

        filtered_agents = filter_agents(agent_trajectories, reverse=True)

        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0]), len(agent_trajectories[-1]))
        self.assertEqual(len(filtered_agents[5]), num_agents - missing_agents)
        self.assertEqual(len(filtered_agents[-1]), num_agents - missing_agents)


if __name__ == '__main__':
    unittest.main()
