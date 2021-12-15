from typing import Dict, List, Tuple

from nuplan.common.actor_state.agent import Agent, AgentType
from nuplan.common.actor_state.utils import lazy_property


class TrackedObjects:
    def __init__(self, agents: List[Agent]):
        """
        :param agents: List of Agents
        """
        self.agents = sorted(agents, key=lambda a: a.agent_type.value)

    @lazy_property
    def ranges_per_type(self) -> Dict[AgentType, Tuple[int, int]]:
        """
        Returns the start and end index of the range of agents for each agent type
        in the list of agents (sorted by agent type).
        """
        return self._create_ranges_per_type()

    def _create_ranges_per_type(self) -> Dict[AgentType, Tuple[int, int]]:
        """ Extracts the start and end index of each agent type in the list of agents (sorted by agent type). """
        ranges_per_type: Dict[AgentType, Tuple[int, int]] = {}

        if self.agents:
            last_agent_type = self.agents[0].agent_type
            start_range = 0
            end_range = len(self.agents)

            for idx, agent in enumerate(self.agents):
                if agent.agent_type is not last_agent_type:
                    ranges_per_type[last_agent_type] = (start_range, idx)
                    start_range = idx
                    last_agent_type = agent.agent_type
            ranges_per_type[last_agent_type] = (start_range, end_range)

            ranges_per_type.update(
                {agent_type: (end_range, end_range) for agent_type in AgentType if agent_type not in ranges_per_type})

        return ranges_per_type

    def get_agents_of_type(self, agent_type: AgentType) -> List[Agent]:
        """
        Gets the sublist of agents of a particular AgentType
        :param agent_type: The query AgentType
        :return: List of the present agents of the query type. Throws an error if the key is invalid. """
        if agent_type in self.ranges_per_type:
            start_idx, end_idx = self.ranges_per_type[agent_type]
            return self.agents[start_idx:end_idx]

        else:
            raise KeyError(f"Specified AgentType {agent_type} does not exist.")
