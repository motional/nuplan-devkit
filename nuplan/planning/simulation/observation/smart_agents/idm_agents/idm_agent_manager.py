from typing import Dict

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_agent import IDMAgent
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_states import IDMLeadAgentState
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import box3d_to_polygon, \
    ego_state_to_box_3d, path_to_linestring, rotate_vector
from nuplan.planning.simulation.observation.smart_agents.occupancy_map.abstract_occupancy_map import OccupancyMap
from shapely.geometry.base import CAP_STYLE

UniqueIDMAgents = Dict[str, IDMAgent]


class IDMAgentManager:

    def __init__(self, agents: UniqueIDMAgents, agent_occupancy: OccupancyMap):
        self.agents: UniqueIDMAgents = agents
        self.agent_occupancy = agent_occupancy

    def propagate_agents(self, ego_state: EgoState, tspan: float, iteration: int) -> None:
        """
        Propagate each active agent forward in time.

        :param ego_state: the ego's current state in the simulation
        :param tspan: the interval of time to simulate
        :param iteration: the simulation iteration
        """
        ego_box = ego_state_to_box_3d(ego_state)
        self.agent_occupancy.set("ego", box3d_to_polygon(ego_box))
        for agent_token, agent in self.agents.items():
            if agent.is_active(iteration) and agent.has_valid_path() is not None:

                # Check for agents that intersects THIS agent's path
                agent_path = path_to_linestring(agent.get_path_to_go())
                intersecting_agents = self.agent_occupancy.intersects(agent_path.buffer((agent.width / 2),
                                                                                        cap_style=CAP_STYLE.flat))
                assert intersecting_agents.contains(
                    agent_token), "Agent's baseline does not intersect the agent itself"

                # Checking if there are agents intersecting THIS agent's baseline.
                # Hence, we are checking for at least 2 intersecting agents
                if intersecting_agents.size > 1:
                    nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(
                        agent_token)
                    agent_heading = agent.to_se2().heading
                    if nearest_id == "ego":
                        longitudinal_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
                        relative_heading = ego_state.rear_axle.heading - agent_heading
                    else:
                        nearest_agent = self.agents[nearest_id]
                        longitudinal_velocity = nearest_agent.velocity
                        relative_heading = nearest_agent.to_se2().heading - agent_heading

                    # Wrap angle to [-pi, pi]
                    relative_heading = np.arctan2(np.sin(relative_heading), np.cos(relative_heading))

                    # take the longitudinal component of the projected velocity
                    projected_velocity = rotate_vector((longitudinal_velocity, 0, 0), relative_heading)[0]

                    # relative_distance already takes the vehicle dimension into account.
                    # Therefore there is no need to pass in the length_rear
                    length_rear = 0
                else:
                    # Free road case: no leading vehicle
                    projected_velocity = 0.0
                    relative_distance = agent.get_progress_to_go()
                    length_rear = agent.length / 2

                agent.propagate(IDMLeadAgentState(progress=relative_distance,
                                                  velocity=projected_velocity,
                                                  length_rear=length_rear),
                                tspan)

                self.agent_occupancy.set(agent_token, agent.polygon)

    def get_active_agents(self, iteration: int, num_samples: int, sampling_time: float) -> Detections:
        """
        Returns all agents as Detections
        :param iteration: the current simulation iteration
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: agents as Detections
        """
        return Detections([agent.get_box_with_planned_trajectory(num_samples, sampling_time)
                           for agent in self.agents.values() if agent.is_active(iteration)])
