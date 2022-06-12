import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


def ego_pose_to_array(ego_pose: EgoState) -> npt.NDArray[np.float32]:
    """
    Convert EgoState to array
    :param ego_pose: agent state
    :return: [x, y, heading]
    """
    return np.array([ego_pose.rear_axle.x, ego_pose.rear_axle.y, ego_pose.rear_axle.heading])


def extract_initial_offset(scenario: AbstractScenario) -> npt.NDArray[np.float32]:
    """
    Return offset
    :param scenario: for which offset should be computed
    :return: offset of type [x, y, heading]
    """
    initial_ego_pose = ego_pose_to_array(scenario.get_ego_state_at_iteration(0))
    initial_offset: npt.NDArray[np.float32] = np.array([initial_ego_pose[0], initial_ego_pose[1], 0.0])

    return initial_offset


def extract_ego_trajectory(
    scenario: AbstractScenario,
    shorten_end_of_scenario: int,
    offset_start_of_scenario: int = 0,
    subtract_initial_pose_offset: bool = True,
) -> Trajectory:
    """
    Extract ego trajectory from scenario
    :param scenario: for which ego trajectory should be extracted
    :param shorten_end_of_scenario: future poses
    :param offset_start_of_scenario: future poses
    :param subtract_initial_pose_offset: subtract offset from initial ego pose
    :return: Ego Trajectory
    """
    # Create default scene
    iteration_end = scenario.get_number_of_iterations() - shorten_end_of_scenario
    iteration_start = offset_start_of_scenario
    number_of_scenes = iteration_end - iteration_start

    trajectory_poses = np.zeros((number_of_scenes, Trajectory.state_size()))

    for index, index_in_scenario in enumerate(np.arange(iteration_start, iteration_end)):
        ego_pose = scenario.get_ego_state_at_iteration(index_in_scenario)
        trajectory_poses[index] = ego_pose_to_array(ego_pose)

    # Create initial offset.
    if subtract_initial_pose_offset:
        initial_offset = extract_initial_offset(scenario)
        trajectory_poses = trajectory_poses - initial_offset

    return Trajectory(data=trajectory_poses.astype(np.float32))
