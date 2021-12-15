from typing import List, Optional

import numpy as np
from nuplan.common.actor_state.state_representation import ProgressStateSE2, StateSE2
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.geometry import yaw_to_quaternion
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_policy import IDMPolicy
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_states import IDMAgentState, IDMLeadAgentState
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import box3d_to_polygon
from nuplan.planning.simulation.path.interpolated_path import AbstractPath
from nuplan.planning.simulation.path.utils import get_trimmed_path_up_to_progress
from shapely.geometry import Polygon


class IDMAgent:

    def __init__(self, start_iteration: int,
                 intial_state: Box3D,
                 path: AbstractPath,
                 path_progress: float,
                 policy: IDMPolicy):
        """
        Constructor for IDMAgent
        :param start_iteration: scenario iteration where agent first appeared
        :param intial_state: agent initial state
        :param path: agent initial state
        """

        self._start_iteration = start_iteration  # scenario iteration where agent first appears
        self._state: IDMAgentState = IDMAgentState(path_progress, intial_state.velocity[0])
        self._initial_state = intial_state
        self._path: AbstractPath = path
        self._policy: IDMPolicy = policy
        self._size = intial_state.size

    def propagate(self, lead_agent: IDMLeadAgentState, tspan: float) -> None:
        """
        Propagate agent forward according to the IDM policy

        :param lead_agent: the agent leading this agent
        :param tspan: the interval of time to propagate for
        """

        solution = self._policy.solve_forward_euler_idm_policy(
            IDMAgentState(0, self._state.velocity), lead_agent, tspan)
        self._state.progress += solution.progress
        self._state.velocity = max(solution.velocity, 0)

    @property
    def box(self) -> Box3D:
        """ :return: the agent as a Box3D object """
        return self._get_box_at_progress(self._get_bounded_progress())

    @property
    def polygon(self) -> Polygon:
        """ :return: the agent as a Box3D object """
        return box3d_to_polygon(self.box)

    @property
    def width(self) -> float:
        """ :return: [m] agent's width  """
        return float(self._size[0])

    @property
    def length(self) -> float:
        """ :return: [m] agent's length """
        return float(self._size[1])

    @property
    def progress(self) -> float:
        """ :return: [m] agent's progress """
        return self._state.progress  # type: ignore

    @property
    def velocity(self) -> float:
        """ :return: [m/s] agent's velocity along the path"""
        return self._state.velocity  # type: ignore

    def to_se2(self) -> StateSE2:
        """
        :return: the agent as a StateSE2 object
        """
        pose = self.get_agent_progress_state()
        return StateSE2(x=pose.x, y=pose.y, heading=pose.heading)

    def is_active(self, iteration: int) -> bool:
        """
        Return if the agent should be active at a simulation iteration

        :param iteration: the current simulation iteration
        :return: true if active, false otherwise
        """
        return self._start_iteration <= iteration

    def has_valid_path(self) -> bool:
        """
        :return: true if agent has a valid path, false otherwise
        """
        return self._path is not None

    def get_agent_progress_state(self) -> ProgressStateSE2:
        """
        :return: agent pose at the current progress along the agent's path
        """
        progress = self._get_bounded_progress()
        return self._path.get_state_at_progress(progress)

    def _get_bounded_progress(self) -> float:
        """
        :return: [m] The agent's progress. The progress is clamped between the start and end progress of it's path
        """
        return self._clamp_progress(self._state.progress)

    def get_path_to_go(self) -> List[ProgressStateSE2]:
        """
        :return: The agent's path trimmed to start at the agent's current progress
        """
        return get_trimmed_path_up_to_progress(self._path, self._get_bounded_progress())  # type: ignore

    def get_progress_to_go(self) -> float:
        """
        return: [m] the progress left until the end of the path
        """
        return self._path.get_end_progress() - self.progress  # type: ignore

    def get_box_with_planned_trajectory(self, num_samples: int, sampling_time: float) -> Box3D:
        """
        Samples the the agent's trajectory. The velocity is assumed to be constant over the sampled trajectory
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: the agent's trajectory as a list of Box3D
        """
        return self._get_box_at_progress(self._get_bounded_progress(), num_samples, sampling_time)

    def _get_box_at_progress(self, progress: float,
                             num_samples: Optional[int] = None,
                             sampling_time: Optional[float] = None) -> Box3D:
        """
        Returns the agent as a box at a given progress
        :param progress: the arc length along the agent's path
        :return: the agent as a Box3D object at the given progress
        """

        if self._path is not None:
            future_horizon_len_s = None
            future_interval_s = None
            future_centers = None
            future_orientations = None
            mode_probs = None

            progress = self._clamp_progress(progress)
            init_pose = self._path.get_state_at_progress(progress)
            init_orientation = yaw_to_quaternion(init_pose.heading)

            if num_samples is not None and sampling_time is not None:
                progress_samples = [self._clamp_progress(progress + self.velocity * sampling_time * (step + 1))
                                    for step in range(num_samples)]
                poses = [self._path.get_state_at_progress(progress) for progress in progress_samples]
                future_horizon_len_s = num_samples * sampling_time
                future_interval_s = sampling_time
                future_centers = [[(pose.x, pose.y, 0.0) for pose in poses]]
                future_orientations = [[yaw_to_quaternion(pose.heading)for pose in poses]]
                mode_probs = [1.0]

            return Box3D(center=(init_pose.x, init_pose.y, 0),
                         size=self._size,
                         orientation=init_orientation,
                         velocity=(self._state.velocity * np.cos(init_pose.heading),
                                   self._state.velocity * np.sin(init_pose.heading), 0),
                         label=self._initial_state.label,
                         future_horizon_len_s=future_horizon_len_s,
                         future_interval_s=future_interval_s,
                         future_centers=future_centers,
                         future_orientations=future_orientations,
                         mode_probs=mode_probs)
        return self._initial_state

    def _clamp_progress(self, progress: float) -> float:
        """
        Clamp the progress to be between the agent's path bounds
        :param progress: [m] the progress along the agent's path
        :return: [m] the progress clamped between the start and end progress of the agent's path
        """
        return max(self._path.get_start_progress(), (min(progress, self._path.get_end_progress())))  # type: ignore
