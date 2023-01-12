import unittest
from typing import List

import numpy as np
import torch

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects, TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentFeatureIndex,
    AgentInternalIndex,
    EgoFeatureIndex,
    EgoInternalIndex,
    GenericEgoFeatureIndex,
    build_ego_features,
    build_ego_features_from_tensor,
    build_generic_ego_features_from_tensor,
    compute_yaw_rate_from_state_tensors,
    compute_yaw_rate_from_states,
    convert_absolute_quantities_to_relative,
    extract_and_pad_agent_poses,
    extract_and_pad_agent_sizes,
    extract_and_pad_agent_velocities,
    filter_agents,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list,
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


def _create_ego_trajectory_tensor(num_frames: int) -> torch.Tensor:
    """
    Generate a dummy ego trajectory
    :param num_frames: length of the trajectory to be generate
    :return: The generated trajectory
    """
    output = torch.ones((num_frames, EgoInternalIndex.dim()))
    for i in range(num_frames):
        output[i, :] *= i

    return output


def _create_tracked_object_agent_tensor(num_agents: int) -> torch.Tensor:
    """
    Generates a dummy tracked object input tensor.
    :param num_agents: The number of agents in the tensor.
    :return: The generated tensor.
    """
    output = torch.ones((num_agents, AgentInternalIndex.dim()))
    for i in range(num_agents):
        output[i, :] *= i

    return output


def _create_dummy_tracked_objects_tensor(num_frames: int) -> List[TrackedObjects]:
    """
    Generates some dummy tracked objects for use with testing the tensorization functions.
    :param num_frames: The number of frames for which to generate the objects.
    :return: The generated dummy objects.
    """
    test_tracked_objects = []
    for i in range(num_frames):
        num_agents_in_frame = i + 1
        num_non_agents_in_frame = num_frames + 1 - num_agents_in_frame

        objects_in_frame: List[TrackedObject] = []
        for j in range(num_agents_in_frame):
            # The numbers here are selected to ensure the correct fields are picked.
            #
            # If all goes well, we will expect a tensor generated from these objects to look like
            #
            # 0, 1, 2, 3, 4, 5, 6, 7, 8
            # 1, 2, 3, 4, 5, 6, 7, 8, 9
            # ...
            #
            # Common errors:
            #   -1 => the wrong field was read
            #   Numbers 100+ => static objects not filtered.
            objects_in_frame.append(
                Agent(
                    tracked_object_type=TrackedObjectType.VEHICLE,
                    oriented_box=OrientedBox(
                        center=StateSE2(x=j + 6, y=j + 7, heading=j + 3),
                        length=j + 5,
                        width=j + 4,
                        height=-1,
                    ),
                    velocity=StateVector2D(x=j + 1, y=j + 2),
                    metadata=SceneObjectMetadata(
                        timestamp_us=1, token=f"agent_{j}", track_id=f"agent_{j}", track_token=f"agent_{j}"
                    ),
                    angular_velocity=-1,
                    predictions=None,
                    past_trajectory=None,
                )
            )

            objects_in_frame.append(
                Agent(
                    tracked_object_type=TrackedObjectType.BICYCLE,
                    oriented_box=OrientedBox(
                        center=StateSE2(x=j + 6, y=j + 7, heading=j + 3),
                        length=j + 5,
                        width=j + 4,
                        height=-1,
                    ),
                    velocity=StateVector2D(x=j + 1, y=j + 2),
                    metadata=SceneObjectMetadata(
                        timestamp_us=1, token=f"agent_{j}", track_id=f"agent_{j}", track_token=f"agent_{j}"
                    ),
                    angular_velocity=-1,
                    predictions=None,
                    past_trajectory=None,
                )
            )

            objects_in_frame.append(
                Agent(
                    tracked_object_type=TrackedObjectType.PEDESTRIAN,
                    oriented_box=OrientedBox(
                        center=StateSE2(x=j + 6, y=j + 7, heading=j + 3),
                        length=j + 5,
                        width=j + 4,
                        height=-1,
                    ),
                    velocity=StateVector2D(x=j + 1, y=j + 2),
                    metadata=SceneObjectMetadata(
                        timestamp_us=1, token=f"agent_{j}", track_id=f"agent_{j}", track_token=f"agent_{j}"
                    ),
                    angular_velocity=-1,
                    predictions=None,
                    past_trajectory=None,
                )
            )

        for j in range(num_non_agents_in_frame):
            jj = j + 100
            objects_in_frame.append(
                StaticObject(
                    tracked_object_type=TrackedObjectType.GENERIC_OBJECT,
                    oriented_box=OrientedBox(
                        center=StateSE2(x=jj, y=jj, heading=jj),
                        length=jj,
                        width=jj,
                        height=jj,
                    ),
                    metadata=SceneObjectMetadata(
                        timestamp_us=jj, token=f"static_{jj}", track_id=f"static_{jj}", track_token=f"static_{jj}"
                    ),
                )
            )

        test_tracked_objects.append(TrackedObjects(objects_in_frame))

    return test_tracked_objects


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

        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(np.allclose(ego_features[0], np.array([0, 0, 0])))

        ego_features_reversed = build_ego_features(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features_reversed.shape)
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

        tracked_objects_history = [
            *_create_tracked_objects(num_frames=5, num_agents=num_agents, object_type=TrackedObjectType.BICYCLE),
            *_create_tracked_objects(
                num_frames=2, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.VEHICLE
            ),
            *_create_tracked_objects(
                num_frames=1, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.BICYCLE
            ),
        ]
        filtered_agents = filter_agents(tracked_objects_history, allowable_types=[TrackedObjectType.BICYCLE])
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[0].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)

    def test_build_ego_features_from_tensor(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        zeros = torch.tensor([0, 0, 0], dtype=torch.float32)
        ego_trajectory = _create_ego_trajectory_tensor(num_frames)
        ego_features = build_ego_features_from_tensor(ego_trajectory)

        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(torch.allclose(ego_features[0], zeros, atol=1e-7))

        ego_features_reversed = build_ego_features_from_tensor(ego_trajectory, reverse=True)

        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(torch.allclose(ego_features_reversed[-1], zeros, atol=1e-7))

    def test_build_generic_ego_features_from_tensor(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        ego_trajectory = _create_ego_trajectory_tensor(num_frames)
        ego_features = build_generic_ego_features_from_tensor(ego_trajectory)

        self.assertEqual((num_frames, GenericEgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(torch.allclose(ego_features[0], zeros, atol=1e-7))

        ego_features_reversed = build_generic_ego_features_from_tensor(ego_trajectory, reverse=True)

        self.assertEqual((num_frames, GenericEgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(torch.allclose(ego_features_reversed[-1], zeros, atol=1e-7))

    def test_convert_absolute_quantities_to_relative(self) -> None:
        """
        Test the conversion routine between absolute and relative quantities
        """

        def get_dummy_states() -> List[torch.Tensor]:
            """
            Create a series of dummy agent tensors
            """
            # Create tensors that look like
            # 0, 0, ..., 0    1, 1, ..., 1
            # 1, 1, ..., 1  , 2, 2, ..., 2
            # ...             ...
            # 4, 4, ..., 4    5, 5, ..., 5
            dummy_agent_state = _create_tracked_object_agent_tensor(7)
            dummy_states = [dummy_agent_state + i for i in range(5)]
            return dummy_states

        zeros = torch.tensor([0, 0, 0], dtype=torch.float32)

        # First sanity check: that pose is transformed correctly
        #
        # In each tensor, there will be a row where the pose of [x, y, h] is [4, 4, 4]
        # Check that that row gets transformed to [0, 0, 0]
        # Fill other values (e.g. velocity) with junk data so it becomes obvious if the wrong field is used.
        dummy_states = get_dummy_states()
        ego_pose = torch.tensor([4, 4, 4, 2, 2, 2, 2], dtype=torch.float32)
        transformed = convert_absolute_quantities_to_relative(dummy_states, ego_pose)

        for i in range(0, len(transformed), 1):
            should_be_zero_row = 4 - i
            check_tensor = torch.tensor(
                [
                    transformed[i][should_be_zero_row, AgentInternalIndex.x()].item(),
                    transformed[i][should_be_zero_row, AgentInternalIndex.y()].item(),
                    transformed[i][should_be_zero_row, AgentInternalIndex.heading()].item(),
                ],
                dtype=torch.float32,
            )
            self.assertTrue(torch.allclose(check_tensor, zeros, atol=1e-7))

        # Second sanity check: that velocity is transformed correctly
        #
        # In each tensor, there will be a row where the velocity of [vx, vy, h] is [4, 4, 4]
        # Check that that row gets transformed to [0, 0, 0]
        # Fill other values (e.g. velocity) with junk data so it becomes obvious if the wrong field is used.
        dummy_states = get_dummy_states()
        ego_pose = torch.tensor([2, 2, 4, 4, 4, 4, 4], dtype=torch.float32)
        transformed = convert_absolute_quantities_to_relative(dummy_states, ego_pose)

        for i in range(0, len(transformed), 1):
            should_be_zero_row = 4 - i
            check_tensor = torch.tensor(
                [
                    transformed[i][should_be_zero_row, AgentInternalIndex.vx()].item(),
                    transformed[i][should_be_zero_row, AgentInternalIndex.vy()].item(),
                    transformed[i][should_be_zero_row, AgentInternalIndex.heading()].item(),
                ],
                dtype=torch.float32,
            )
            self.assertTrue(torch.allclose(check_tensor, zeros, atol=1e-7))

    def test_pad_agent_states(self) -> None:
        """
        Test the pad agent states functionality
        """
        # First test - missing agents in later frames.
        # Check that the second and third states get properly padded.
        forward_dummy_states = [
            _create_tracked_object_agent_tensor(7),
            _create_tracked_object_agent_tensor(5),
            _create_tracked_object_agent_tensor(6),
        ]

        padded = pad_agent_states(forward_dummy_states, reverse=False)

        self.assertTrue(len(padded) == 3)
        self.assertEqual((7, AgentInternalIndex.dim()), padded[0].shape)
        for i in range(1, len(padded)):
            self.assertTrue(torch.allclose(padded[0], padded[i]))

        # Second test - missing agents in final frame
        # Check that the first and second states get properly padded.
        backward_dummy_states = [
            _create_tracked_object_agent_tensor(6),
            _create_tracked_object_agent_tensor(5),
            _create_tracked_object_agent_tensor(7),
        ]

        padded_reverse = pad_agent_states(backward_dummy_states, reverse=True)

        self.assertTrue(len(padded_reverse) == 3)
        self.assertEqual((7, AgentInternalIndex.dim()), padded_reverse[2].shape)
        for i in range(0, len(padded_reverse) - 1):
            self.assertTrue(torch.allclose(padded_reverse[2], padded_reverse[i]))

    def test_compute_yaw_rate_from_state_tensors(self) -> None:
        """
        Test compute yaw rate functionality
        """
        num_frames = 6
        num_agents = 5

        agent_states = [_create_tracked_object_agent_tensor(num_agents) + i for i in range(num_frames)]
        time_stamps = torch.tensor([int(i * 1e6) for i in range(num_frames)], dtype=torch.int64)

        yaw_rate = compute_yaw_rate_from_state_tensors(agent_states, time_stamps)
        self.assertEqual((num_frames, num_agents), yaw_rate.shape)
        self.assertTrue(torch.allclose(torch.ones((num_frames, num_agents), dtype=torch.float64), yaw_rate))

    def test_filter_agents_tensor(self) -> None:
        """
        Test filter agents
        """
        # First test. Check that the extra agents in the second frame are filtered.
        dummy_states = [
            _create_tracked_object_agent_tensor(7),
            _create_tracked_object_agent_tensor(8),
            _create_tracked_object_agent_tensor(6),
        ]

        filtered = filter_agents_tensor(dummy_states, reverse=False)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered[0].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered[1].shape)
        self.assertEqual((6, AgentInternalIndex.dim()), filtered[2].shape)

        dummy_states = [
            _create_tracked_object_agent_tensor(6),
            _create_tracked_object_agent_tensor(8),
            _create_tracked_object_agent_tensor(7),
        ]

        filtered_reverse = filter_agents_tensor(dummy_states, reverse=True)
        self.assertEqual((6, AgentInternalIndex.dim()), filtered_reverse[0].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered_reverse[1].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered_reverse[2].shape)

    def test_sampled_past_ego_states_to_tensor(self) -> None:
        """
        Test the conversion routine to convert ego states to tensors.
        """
        num_egos = 6
        test_egos = []
        for i in range(num_egos):
            footprint = CarFootprint(
                center=StateSE2(x=i, y=i, heading=i),
                vehicle_parameters=VehicleParameters(
                    vehicle_name="vehicle_name",
                    vehicle_type="vehicle_type",
                    width=i,
                    front_length=i,
                    rear_length=i,
                    cog_position_from_rear_axle=i,
                    wheel_base=i,
                    height=i,
                ),
            )
            dynamic_car_state = DynamicCarState(
                rear_axle_to_center_dist=i,
                rear_axle_velocity_2d=StateVector2D(x=i + 5, y=i + 5),
                rear_axle_acceleration_2d=StateVector2D(x=i, y=i),
                angular_velocity=i,
                angular_acceleration=i,
                tire_steering_rate=i,
            )

            test_ego = EgoState(
                car_footprint=footprint,
                dynamic_car_state=dynamic_car_state,
                tire_steering_angle=i,
                is_in_auto_mode=i,
                time_point=TimePoint(time_us=i),
            )

            test_egos.append(test_ego)

        tensor = sampled_past_ego_states_to_tensor(test_egos)
        self.assertEqual((6, EgoInternalIndex.dim()), tensor.shape)

        for i in range(0, tensor.shape[0], 1):
            ego = test_egos[i]

            self.assertEqual(ego.rear_axle.x, tensor[i, EgoInternalIndex.x()].item())
            self.assertEqual(ego.rear_axle.y, tensor[i, EgoInternalIndex.y()].item())
            self.assertEqual(ego.rear_axle.heading, tensor[i, EgoInternalIndex.heading()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_velocity_2d.x, tensor[i, EgoInternalIndex.vx()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_velocity_2d.y, tensor[i, EgoInternalIndex.vy()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_acceleration_2d.x, tensor[i, EgoInternalIndex.ax()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_acceleration_2d.y, tensor[i, EgoInternalIndex.ay()].item())

    def test_sampled_past_timestamps_to_tensor(self) -> None:
        """
        Test the conversion routine to convert timestamps to tensors.
        """
        points = [TimePoint(time_us=i) for i in range(10)]
        tensor = sampled_past_timestamps_to_tensor(points)

        self.assertEqual((10,), tensor.shape)
        for i in range(tensor.shape[0]):
            self.assertEqual(i, int(tensor[i].item()))

    def test_tracked_objects_to_tensor_list(self) -> None:
        """
        Test the conversion routine to convert tracked objects to tensors.
        """
        num_frames = 5
        test_tracked_objects = _create_dummy_tracked_objects_tensor(num_frames)

        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)

            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))

        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects, object_type=TrackedObjectType.BICYCLE)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)

            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))

        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects, object_type=TrackedObjectType.PEDESTRIAN)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)

            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))

    def test_pack_agents_tensor(self) -> None:
        """
        Test the routine used to convert local buffers into the final feature.
        """
        num_agents = 4
        num_timestamps = 3

        agents_tensors = [_create_tracked_object_agent_tensor(num_agents) for _ in range(num_timestamps)]

        yaw_rates = torch.ones((num_timestamps, num_agents)) * 100

        packed = pack_agents_tensor(agents_tensors, yaw_rates)

        self.assertEqual((num_timestamps, num_agents, AgentFeatureIndex.dim()), packed.shape)
        for ts in range(num_timestamps):
            for agent in range(num_agents):
                for col in range(AgentFeatureIndex.dim()):
                    if col == AgentFeatureIndex.yaw_rate():
                        self.assertEqual(100, packed[ts, agent, col])
                    else:
                        self.assertEqual(agent, packed[ts, agent, col])


if __name__ == '__main__':
    unittest.main()
