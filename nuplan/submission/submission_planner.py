import logging
import os
from concurrent import futures

import grpc
from omegaconf import DictConfig

from nuplan.common.maps.map_manager import MapManager
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.submission import challenge_pb2_grpc as chpb_grpc
from nuplan.submission.challenge_servicers import DetectionTracksChallengeServicer

logger = logging.getLogger(__name__)


class SubmissionPlanner:
    """
    Class holding a planner and exposing functionalities as a server. The services are planner initialization and
    trajectory computation.
    """

    def __init__(self, planner_config: DictConfig):
        """
        Prepares the planner and the server. The communication port is read from an environmental variable.
        :param planner_config: The planner configuration to instantiate the planner
        """
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        map_version = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
        map_factory = NuPlanMapFactory(
            GPKGMapsDB(
                map_version=map_version,
                map_root=os.path.join(os.getenv('NUPLAN_DATA_ROOT', "~/nuplan/dataset"), 'maps'),
            )
        )
        map_manager = MapManager(map_factory)
        chpb_grpc.add_DetectionTracksChallengeServicer_to_server(
            DetectionTracksChallengeServicer(planner_config, map_manager), self.server
        )

        port = os.getenv("SUBMISSION_CONTAINER_PORT", 50051)
        logger.info(f"Submission container starting with port {port}")
        if not port:
            raise RuntimeError("Environment variable not specified: 'SUBMISSION_CONTAINER_PORT'")
        self.server.add_insecure_port(f'[::]:{port}')

    def serve(self) -> None:
        """Starts the server."""
        logger.info("Server starting...")

        self.server.start()

        logger.info("Server started!")

        self.server.wait_for_termination()

        logger.info("Server terminated!")
