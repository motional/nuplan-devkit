from __future__ import annotations

import logging

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.remote_planner import RemotePlanner
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.submission.submission_container_factory import SubmissionContainerFactory
from nuplan.submission.submission_container_manager import SubmissionContainerManager
from nuplan.submission.utils.utils import container_name_from_image_name
from nuplan.submission.validators.base_submission_validator import BaseSubmissionValidator

logger = logging.getLogger(__name__)


class SubmissionComputesTrajectoryValidator(BaseSubmissionValidator):
    """Checks if a submission is able to compute a trajectory"""

    def validate(self, submission: str) -> bool:
        """
        Checks if the provided submission is able to provide a trajectory given an input.
        :param submission: The submission image
        :return: Whether the submission is able to compute a trajectory
        """
        logger.info("validating trajectory computation")
        scenario = get_test_nuplan_scenario()

        step = 0
        iteration = SimulationIteration(time_point=scenario.get_time_point(0), index=0)
        history = SimulationHistoryBuffer.initialize_from_list(
            1,
            [scenario.get_ego_state_at_iteration(step)],
            [scenario.get_tracked_objects_at_iteration(step)],
            scenario.database_interval,
        )
        planner_input = PlannerInput(iteration, history)

        # Initialize planner inside container
        container_name = container_name_from_image_name(submission)
        container_manager = SubmissionContainerManager(SubmissionContainerFactory())
        planner = RemotePlanner(container_manager, submission, container_name)
        planner_initialization = PlannerInitialization(
            scenario.get_mission_goal(),
            scenario.get_route_roadblock_ids(),
            scenario.map_api,
        )
        planner.initialize([planner_initialization])

        # Call compute trajectory service
        trajectory = planner.compute_trajectory([planner_input])

        if trajectory:
            logger.debug(f"Computed trajectory {trajectory}")
            return True

        logger.error("Submission failed to compute trajectory")
        self._failing_validator = SubmissionComputesTrajectoryValidator

        return False
