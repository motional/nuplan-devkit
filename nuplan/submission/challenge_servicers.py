import logging
import pickle
from typing import Any, Optional

from omegaconf import DictConfig

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.maps.map_manager import MapManager
from nuplan.planning.script.builders.planner_builder import build_planners
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

    def __init__(
        self,
        planner_config: DictConfig,
        map_manager: MapManager,
    ):
        """
        :param planner_config: The planner configuration to instantiate the planner.
        :param map_manager: The map manager.
        """
        self.planner: Optional[AbstractPlanner] = None
        self._planner_config = planner_config
        self.map_manager = map_manager
        self.simulation_history_buffer: Optional[SimulationHistoryBuffer] = None
        self._initialized = False

    @staticmethod
    def _extract_simulation_iteration(planner_input_message: chpb.PlannerInput) -> SimulationIteration:
        return SimulationIteration(
            TimePoint(planner_input_message.simulation_iteration.time_us),
            planner_input_message.simulation_iteration.index,
        )

    def _build_planner_input(
        self, planner_input_message: chpb.PlannerInput, buffer: Optional[SimulationHistoryBuffer]
    ) -> PlannerInput:
        """
        Builds a PlannerInput from a serialized PlannerInput message and an existing data buffer
        :param planner_input_message: the serialized message
        :param buffer: The history buffer
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
            self.simulation_history_buffer = buffer

        tl_data_messages = planner_input_message.traffic_light_data
        tl_data = [tl_status_data_from_proto_tl_status_data(tl_data_message) for tl_data_message in tl_data_messages]

        return PlannerInput(iteration=simulation_iteration, history=buffer, traffic_light_data=tl_data)

    def InitializePlanner(
        self, planner_initialization_message: chpb.PlannerInitializationLight, context: Any
    ) -> chpb.Empty:
        """
        Service to initialize the planner given the initialization request.
        :param planner_initialization_message: Message containing initialization details
        :param context
        """
        planners = build_planners(self._planner_config, None)
        assert len(planners) == 1, f"Configuration should build exactly 1 planner, got {len(planners)} instead!"
        self.planner = planners[0]

        logger.info("Initialization request received..")

        route_roadblock_ids = planner_initialization_message.route_roadblock_ids
        mission_goal = se2_from_proto_se2(planner_initialization_message.mission_goal)

        map_api = self.map_manager.get_map(planner_initialization_message.map_name)
        map_api.initialize_all_layers()
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=route_roadblock_ids,
            mission_goal=mission_goal,
            map_api=map_api,
        )

        self.simulation_history_buffer = None

        self.planner.initialize(planner_initialization)

        logging.info("Planner initialized!")
        self._initialized = True
        return chpb.Empty()

    def ComputeTrajectory(self, planner_input_message: chpb.PlannerInput, context: Any) -> chpb.Trajectory:
        """
        Service to compute a trajectory given a planner input message
        :param planner_input_message: Message containing the input to the planner
        :param context
        :return Message containing the computed trajectories
        """
        assert self._initialized, "Planner has not been initialized. Please call InitializePlanner"

        planner_inputs = self._build_planner_input(planner_input_message, self.simulation_history_buffer)

        if isinstance(self.planner, AbstractPlanner):
            trajectory = self.planner.compute_trajectory(planner_inputs)

            return proto_traj_from_inter_traj(trajectory)

        raise RuntimeError("The planner was not initialized correctly!")
