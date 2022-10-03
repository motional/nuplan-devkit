from typing import List

from shapely.geometry import LineString

from nuplan.planning.scenario_builder.test.mock_abstract_scenario_builder import MockAbstractScenario
from nuplan.planning.simulation.observation.idm.utils import create_path_from_ego_state
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from nuplan.planning.simulation.planner.abstract_idm_planner import AbstractIDMPlanner
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class MockIDMPlanner(AbstractIDMPlanner):
    """
    Mock IDMPlanner class for testing the AbstractIDMPlanner interface
    """

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
        planned_trajectory_samples: int,
        planned_trajectory_sample_interval: float,
        occupancy_map_radius: float,
    ):
        """
        Constructor for IDMPlanner
        :param target_velocity: [m/s] Desired velocity in free traffic.
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle.
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front.
        :param accel_max: [m/s^2] maximum acceleration.
        :param decel_max: [m/s^2] maximum deceleration (positive value).
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param occupancy_map_radius: [m] The range around the ego to add objects to be considered.
        """
        super(MockIDMPlanner, self).__init__(
            target_velocity,
            min_gap_to_lead_agent,
            headway_time,
            accel_max,
            decel_max,
            planned_trajectory_samples,
            planned_trajectory_sample_interval,
            occupancy_map_radius,
        )

        self._scenario = MockAbstractScenario()
        self._scenario_buffer = 10  # [s] Mock scenario length

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        self._map_api = initialization[0].map_api
        self._initialize_route_plan(initialization[0].route_roadblock_ids)
        self._ego_path: InterpolatedPath = create_path_from_ego_state(
            self._scenario.get_ego_future_trajectory(0, self._scenario_buffer, 10)
        )
        self._ego_path_linestring = LineString()

    def compute_planner_trajectory(self, current_input: List[PlannerInput]) -> List[AbstractTrajectory]:
        """Inherited, see superclass."""
        return [InterpolatedTrajectory(self._ego_path.get_sampled_path())]
