from __future__ import annotations

import logging
import os
import pickle
from typing import List, Optional, Tuple, Type

import docker.errors
import grpc

from nuplan.common.utils.helpers import keep_trying, try_n_times
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.submission import challenge_pb2 as chpb
from nuplan.submission import challenge_pb2_grpc as chpb_grpc
from nuplan.submission.proto_converters import interp_traj_from_proto_traj, proto_se2_from_se2
from nuplan.submission.submission_container_manager import SubmissionContainerManager
from nuplan.submission.utils import find_free_port_number

logger = logging.getLogger(__name__)


class RemotePlanner(AbstractPlanner):
    """
    Remote planner delegates computation of trajectories to a docker container, with which communicates through
    grpc.
    """

    def __init__(
        self,
        submission_container_manager: Optional[SubmissionContainerManager] = None,
        submission_image: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> None:
        """
        Prepares the remote container for planning.
        :param submission_container_manager: Optional manager, if provided a container will be started by RemotePlanner
        :param submission_image: Docker image name for the submission_container_factory
        :param container_name: Name to assign to the submission container
        """
        # If we are requested to start a container, assert all parameters are present
        if submission_container_manager:
            missing_parameter_message = "Parameters for SubmissionContainer are missing!"
            assert submission_image, missing_parameter_message
            assert container_name, missing_parameter_message
            self.port = None
        else:
            self.port = os.getenv("SUBMISSION_CONTAINER_PORT", 50051)

        self.submission_container_manager = submission_container_manager
        self.submission_image = submission_image
        self.container_name = container_name

        self._channel = None
        self._stub = None
        self.serialized_observations: Optional[List[List[bytes]]] = None
        self.serialized_states: Optional[List[List[bytes]]] = None
        self.sample_intervals: Optional[List[float]] = None
        self._consume_batched_inputs = False

    def __reduce__(
        self,
    ) -> Tuple[Type[RemotePlanner], Tuple[Optional[SubmissionContainerManager], Optional[str], Optional[str]]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class
        """
        return self.__class__, (self.submission_container_manager, self.submission_image, self.container_name)

    def name(self) -> str:
        """Inherited, see superclass."""
        return "RemotePlanner"

    @property
    def consume_batched_inputs(self) -> bool:
        """Inherited, see superclass."""
        return self._consume_batched_inputs

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    @staticmethod
    def _planner_initializations_to_message(
        initialization: List[PlannerInitialization],
    ) -> chpb.MultiPlannerInitializationLight:
        """
        Converts a list of PlannerInitialization to the message specified in the protocol files.
        :param initialization: The initialization parameters for the planner
        :return: A initialization message
        """
        planner_initializations = []

        for init in initialization:
            expert_goal_state = proto_se2_from_se2(init.expert_goal_state)
            mission_goal = proto_se2_from_se2(init.mission_goal)
            planner_initializations.append(
                chpb.PlannerInitializationLight(
                    expert_goal_state=expert_goal_state, mission_goal=mission_goal, map_name=init.map_api.map_name
                )
            )

        return chpb.MultiPlannerInitializationLight(planner_initializations=planner_initializations)

    def initialize(self, initialization: List[PlannerInitialization], timeout: float = 5) -> None:
        """
        Creates the container manager, and runs the specified docker image. The communication port is created using
        the PID from the ray worker. Sends a request to initialize the remote planner.
        :param initialization: List of PlannerInitialization objects
        :param timeout: for planner initialization
        """
        network = "submission"

        if self.submission_container_manager:
            submission_container = try_n_times(
                self.submission_container_manager.get_submission_container,
                [self.submission_image, self.container_name, find_free_port_number()],
                {},
                (docker.errors.APIError,),
                max_tries=10,
            )
            self.port = submission_container.port

            submission_container.start()
            submission_container.wait_until_running(timeout=5)
            network = "localhost"

        # As we might be running in a multiprocess environment, we need to make sure we have a unique port
        self._channel = grpc.insecure_channel(f'{network}:{self.port}')
        self._stub = chpb_grpc.DetectionTracksChallengeStub(self._channel)

        logger.info("Client sending planner initialization request...")

        planner_initializations_message = self._planner_initializations_to_message(initialization)

        initialization_response, elapsed_time = keep_trying(
            self._stub.InitializePlanner,  # type: ignore
            [planner_initializations_message],
            {},
            errors=(grpc.RpcError,),
            timeout=timeout,
        )

        # The initialization tells us if the RemotePlanner is able to handle batch inputs
        self._consume_batched_inputs = initialization_response.consume_batched_inputs
        logger.info("Planner initialized!")

    def compute_trajectory(self, current_input: List[PlannerInput]) -> List[AbstractTrajectory]:
        """
        Computes the ego vehicle trajectory.
        :param current_input: List of planner inputs for where for each of them trajectory should be computed
            In this case the list represents batched simulations. In case consume_batched_inputs is False
            the list has only single element
        :return: Trajectories representing the predicted ego's position in future for every input iteration
            In case consume_batched_inputs is False, return only a single trajectory in a list.
        """
        logger.debug(f"Client sending planner input: {current_input}")

        trajectories = self._compute_trajectory(self._stub, current_input=current_input)

        return trajectories

    def _compute_trajectory(
        self, stub: chpb_grpc.DetectionTracksChallengeStub, current_input: List[PlannerInput]
    ) -> List[AbstractTrajectory]:
        """
        Sends a request to compute the trajectories given the PlannerInput to the remote planner.
        :param stub:
        :param current_input: List of planner inputs for where for each of them trajectory should be computed
        :return: Trajectories representing the predicted ego's position in future for every input iteration
            In case consume_batched_inputs is False, return only a single trajectory in a list.
        """
        logging.debug(f"Client sending observation of size: {len(current_input)}")

        self.serialized_states, self.serialized_observations, self.sample_intervals = self._get_history_update(
            current_input
        )

        serialized_simulation_iterations = [
            chpb.SimulationIteration(time_us=planner_input.iteration.time_us, index=planner_input.iteration.index)
            for planner_input in current_input
        ]

        if self.sample_intervals:
            serialized_buffers = [
                chpb.SimulationHistoryBuffer(ego_states=state, observations=observation, sample_interval=interval)
                for state, observation, interval in zip(
                    self.serialized_states, self.serialized_observations, self.sample_intervals
                )
            ]
        else:
            serialized_buffers = [
                chpb.SimulationHistoryBuffer(ego_states=state, observations=observation, sample_interval=None)
                for state, observation in zip(self.serialized_states, self.serialized_observations)
            ]
        planner_inputs = [
            chpb.PlannerInput(
                simulation_iteration=simulation_iteration, simulation_history_buffer=simulation_history_buffer
            )
            for simulation_iteration, simulation_history_buffer in zip(
                serialized_simulation_iterations, serialized_buffers
            )
        ]

        trajectory_message = stub.ComputeTrajectory(chpb.MultiPlannerInput(planner_inputs=planner_inputs))
        trajectories = [interp_traj_from_proto_traj(proto_traj) for proto_traj in trajectory_message.trajectories]

        return trajectories

    def _get_history_update(
        self, multi_planner_input: List[PlannerInput]
    ) -> Tuple[List[List[bytes]], List[List[bytes]], Optional[List[float]]]:
        """
        Gets the new states and observations from the input. If no cache is present, the entire history is
        serialized, otherwise just the last element.
        :param multi_planner_input: The inputs for planners
        :return: Tuple with new serialized state and observations.
        """
        # If we have no previous states, we keep all history
        keep_all_history = not self.serialized_states and not self.serialized_observations

        # Initialize the lists
        batch_size = len(multi_planner_input)
        multi_serialized_states: List[List[bytes]] = [[]] * batch_size
        multi_serialized_observations: List[List[bytes]] = [[]] * batch_size
        sample_intervals: List[float] = []

        for i, planner_input in enumerate(multi_planner_input):
            sample_intervals.append(planner_input.history.sample_interval)

            if keep_all_history:
                multi_serialized_states[i] = [pickle.dumps(state) for state in planner_input.history.ego_states]
                multi_serialized_observations[i] = [pickle.dumps(obs) for obs in planner_input.history.observations]
            else:
                last_ego_state, last_observations = planner_input.history.current_state
                multi_serialized_states[i] = [pickle.dumps(last_ego_state)]
                multi_serialized_observations[i] = [pickle.dumps(last_observations)]

        sample_intervals_: Optional[List[float]] = None if not self.sample_intervals else sample_intervals

        return multi_serialized_states, multi_serialized_observations, sample_intervals_
