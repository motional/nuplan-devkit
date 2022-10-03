import logging
import pickle
from typing import Any, List, Optional

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.maps.map_manager import MapManager
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    SimulationIteration,
)
from nuplan.submission import challenge_pb2 as chpb
from nuplan.submission import challenge_pb2_grpc as chpb_grpc
from nuplan.submission.proto_converters import (
    proto_traj_from_inter_traj,
    se2_from_proto_se2,
    tl_status_data_from_proto_tl_status_data,
)

logger = logging.getLogger(__name__)


class DetectionTracksChallengeServicer(chpb_grpc.DetectionTracksChallengeServicer):
    """
    Servicer for exposing initialization and trajectory computation services to the client.
    It keeps a rolling history buffer to avoid unnecessary serialization/deserialization.
    """

    def __init__(self, planner: AbstractPlanner, map_manager: MapManager):
        """
        :param planner: The planner to be used by the service
        :param map_manager: The map manager
        """
        self.planner = planner
        self.map_manager = map_manager
        self.simulation_history_buffers: List[Optional[SimulationHistoryBuffer]] = []

    @staticmethod
    def _extract_simulation_iteration(planner_input_message: chpb.PlannerInput) -> SimulationIteration:
        return SimulationIteration(
            TimePoint(planner_input_message.simulation_iteration.time_us),
            planner_input_message.simulation_iteration.index,
        )

    def _build_planner_input(
        self, planner_input_message: chpb.PlannerInput, buffer: Optional[SimulationHistoryBuffer], idx: int
    ) -> PlannerInput:
        """
        Builds a PlannerInput from a serialized PlannerInput message and an existing data buffer
        :param planner_input_message: the serialized message
        :param buffer: The history buffer
        :param idx: Index of buffer wrt the list of buffers available.
        :return: PlannerInput object
        """
        simulation_iteration = self._extract_simulation_iteration(planner_input_message)

        new_data = planner_input_message.simulation_history_buffer
        states = []
        observations = []

        # Unpacks the state data from the buffer
        for serialized_state, serialized_observation in zip(new_data.ego_states, new_data.observations):
            states.append(pickle.loads(serialized_state))
            observations.append(pickle.loads(serialized_observation))

        # Initialize the buffer if needed
        if buffer is not None:
            buffer.extend(states, observations)
        else:
            buffer = SimulationHistoryBuffer.initialize_from_list(
                len(states), states, observations, new_data.sample_interval
            )
            self.simulation_history_buffers[idx] = buffer

        tl_data_messages = planner_input_message.traffic_light_data
        tl_data = [tl_status_data_from_proto_tl_status_data(tl_data_message) for tl_data_message in tl_data_messages]

        return PlannerInput(iteration=simulation_iteration, history=buffer, traffic_light_data=tl_data)

    def _build_planner_inputs(self, planner_input_messages: List[chpb.PlannerInput]) -> List[PlannerInput]:
        """
        Builds a list of PlannerInput from a list of serialized PlannerInput messages
        :param planner_input_messages: the serialized messages
        :return: List of deserialized PlannerInput objects
        """
        planner_inputs = []

        for i, (planner_input_message, buffer) in enumerate(
            zip(planner_input_messages, self.simulation_history_buffers)
        ):
            planner_inputs.append(self._build_planner_input(planner_input_message, buffer, i))

        return planner_inputs

    def InitializePlanner(
        self, planner_initialization_messages: chpb.MultiPlannerInitializationLight, context: Any
    ) -> chpb.PlannerInitializationResponse:
        """
        Service to initialize the planner given the initialization request.
        :param planner_initialization_messages: Message containing initialization details
        :param context
        """
        logger.info("Initialization request received..")
        planner_initialization = []

        for planner_initialization_message in planner_initialization_messages.planner_initializations:
            route_roadblock_ids = planner_initialization_message.route_roadblock_ids
            mission_goal = se2_from_proto_se2(planner_initialization_message.mission_goal)

            map_api = self.map_manager.get_map(planner_initialization_message.map_name)
            map_api.initialize_all_layers()
            planner_initialization.append(
                PlannerInitialization(
                    route_roadblock_ids=route_roadblock_ids,
                    mission_goal=mission_goal,
                    map_api=map_api,
                )
            )
            self.simulation_history_buffers.append(None)

        self.planner.initialize(planner_initialization)

        initialization_response = chpb.PlannerInitializationResponse(
            consume_batched_inputs=self.planner.consume_batched_inputs
        )

        logging.info("Planner initialized!")

        return initialization_response

    def ComputeTrajectory(self, planner_input_message: chpb.MultiPlannerInput, context: Any) -> chpb.MultiTrajectory:
        """
        Service to compute a trajectory given a planner input message
        :param planner_input_message: Message containing the input to the planner
        :param context
        :return Message containing the computed trajectories
        """
        assert self.simulation_history_buffers, "Planner has not been initialized. Please call InitializePlanner"

        planner_inputs = self._build_planner_inputs(planner_input_message.planner_inputs)

        trajectories = self.planner.compute_trajectory(planner_inputs)
        serialized_trajectories = [proto_traj_from_inter_traj(trajectory) for trajectory in trajectories]

        return chpb.MultiTrajectory(trajectories=serialized_trajectories)
