from __future__ import annotations

import logging
import os
import pickle
from typing import List, Optional, Tuple, Type

import grpc

import docker.errors

from nuplan.common.utils.helpers import keep_trying, try_n_times
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.submission import challenge_pb2 as chpb
from nuplan.submission import challenge_pb2_grpc as chpb_grpc
from nuplan.submission.proto_converters import (
    interp_traj_from_proto_traj,
    proto_se2_from_se2,
    proto_tl_status_data_from_tl_status_data,
)
from nuplan.submission.submission_container_manager import SubmissionContainerManager
from nuplan.submission.utils.utils import find_free_port_number, get_submission_logger

logger = logging.getLogger(__name__)
submission_logger = get_submission_logger(__name__)

NETWORK = 'localhost'


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
        compute_trajectory_timeout: float = 1,
    ) -> None:
        """
        Prepares the remote container for planning.
        :param submission_container_manager: Optional manager, if provided a container will be started by RemotePlanner
        :param submission_image: Docker image name for the submission_container_factory
        :param container_name: Name to assign to the submission container
        :param compute_trajectory_timeout: Timeout for computation of trajectory.
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
        self.serialized_observation: Optional[List[bytes]] = None
        self.serialized_state: Optional[List[bytes]] = None
        self.sample_interval: Optional[float] = None
        self._compute_trajectory_timeout = compute_trajectory_timeout

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

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    @staticmethod
    def _planner_initializations_to_message(
        initialization: PlannerInitialization,
    ) -> chpb.PlannerInitializationLight:
        """
        Converts a PlannerInitialization to the message specified in the protocol files.
        :param initialization: The initialization parameters for the planner
        :return: A initialization message
        """
        try:
            mission_goal = proto_se2_from_se2(initialization.mission_goal)
        except AttributeError as e:
            logger.error("Mission goal was None!")
            raise e

        planner_initialization = chpb.PlannerInitializationLight(
            route_roadblock_ids=initialization.route_roadblock_ids,
            mission_goal=mission_goal,
            map_name=initialization.map_api.map_name,
        )

        return planner_initialization

    def initialize(self, initialization: PlannerInitialization, timeout: float = 5) -> None:
        """
        Creates the container manager, and runs the specified docker image. The communication port is created using
        the PID from the ray worker. Sends a request to initialize the remote planner.
        :param initialization: List of PlannerInitialization objects
        :param timeout: for planner initialization
        """
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

        # As we might be running in a multiprocess environment, we need to make sure we have a unique port
        self._channel = grpc.insecure_channel(f'{NETWORK}:{self.port}')
        self._stub = chpb_grpc.DetectionTracksChallengeStub(self._channel)

        logger.info("Client sending planner initialization request...")

        planner_initializations_message = self._planner_initializations_to_message(initialization)
        logger.info(f"Trying to communicate on port {NETWORK}:{self.port}")

        try:
            _, _ = keep_trying(
                self._stub.InitializePlanner,  # type: ignore
                [planner_initializations_message],
                {},
                errors=(grpc.RpcError,),
                timeout=timeout,
            )
        except Exception as e:
            submission_logger.error("Planner initialization failed!")
            submission_logger.error(e)
            raise e

        logger.info("Planner initialized!")

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Planner input for which trajectory should be computed
        :return: Trajectory representing the predicted ego's position in future for every input iteration
        """
        logger.debug("Client sending planner input: %s" % current_input)

        trajectory = self._compute_trajectory(self._stub, current_input=current_input)

        return trajectory

    def _compute_trajectory(
        self, stub: chpb_grpc.DetectionTracksChallengeStub, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Sends a request to compute the trajectory given the PlannerInput to the remote planner.
        :param stub: Service interface
        :param current_input: Planner input for which a trajectory should be computed.
        :return: Trajectory representing the predicted ego's position in future for every input iteration
        """
        logging.debug("Client sending observation...")

        self.serialized_state, self.serialized_observation, self.sample_interval = self._get_history_update(
            current_input
        )

        serialized_simulation_iteration = chpb.SimulationIteration(
            time_us=current_input.iteration.time_us, index=current_input.iteration.index
        )

        if self.sample_interval:
            serialized_buffer = chpb.SimulationHistoryBuffer(
                ego_states=self.serialized_state,
                observations=self.serialized_observation,
                sample_interval=self.sample_interval,
            )
        else:
            serialized_buffer = chpb.SimulationHistoryBuffer(
                ego_states=self.serialized_state, observations=self.serialized_observation, sample_interval=None
            )

        tl_data = self._build_tl_message_from_planner_input(current_input)

        planner_input = chpb.PlannerInput(
            simulation_iteration=serialized_simulation_iteration,
            simulation_history_buffer=serialized_buffer,
            traffic_light_data=tl_data,
        )

        try:
            trajectory_message = stub.ComputeTrajectory(planner_input, timeout=self._compute_trajectory_timeout)
        except grpc.RpcError as e:
            submission_logger.error("Trajectory computation service failed!")
            submission_logger.error(e)
            raise e

        return interp_traj_from_proto_traj(trajectory_message)

    def _get_history_update(self, planner_input: PlannerInput) -> Tuple[List[bytes], List[bytes], Optional[float]]:
        """
        Gets the new states and observations from the input. If no cache is present, the entire history is
        serialized, otherwise just the last element.
        :param planner_input: The input for planners
        :return: Tuple with new serialized state and observations.
        """
        # If we have no previous states, we keep all history
        keep_all_history = not self.serialized_state and not self.serialized_observation

        if keep_all_history:
            serialized_state = [pickle.dumps(state) for state in planner_input.history.ego_states]
            serialized_observation = [pickle.dumps(obs) for obs in planner_input.history.observations]
        else:
            last_ego_state, last_observations = planner_input.history.current_state
            serialized_state = [pickle.dumps(last_ego_state)]
            serialized_observation = [pickle.dumps(last_observations)]

        sample_interval = planner_input.history.sample_interval if not self.sample_interval else None

        return serialized_state, serialized_observation, sample_interval

    @staticmethod
    def _build_tl_message_from_planner_input(planner_input: PlannerInput) -> chpb.TrafficLightStatusData:

        tl_status_data: List[List[chpb.TrafficLightStatusData]]
        if planner_input.traffic_light_data is None:
            tl_status_data = [[]]
        else:
            tl_status_data = [
                proto_tl_status_data_from_tl_status_data(tl_status_data)
                for tl_status_data in planner_input.traffic_light_data
            ]

        return tl_status_data
