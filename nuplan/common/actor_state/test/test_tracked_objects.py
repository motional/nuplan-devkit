import unittest

from nuplan.common.actor_state.agent import AgentType
from nuplan.common.actor_state.test.test_utils import get_sample_agent
from nuplan.common.actor_state.tracked_objects import TrackedObjects


class TestTrackedObjects(unittest.TestCase):
    def setUp(self) -> None:
        self.agents = [
            get_sample_agent('foo', AgentType.PEDESTRIAN),
            get_sample_agent('bar', AgentType.VEHICLE),
            get_sample_agent('bar_out_the_car', AgentType.PEDESTRIAN),
            get_sample_agent('baz', AgentType.UNKNOWN),
        ]

    def test_construction(self) -> None:
        """ Tests that the object can be created correctly. """
        tracked_objects = TrackedObjects(self.agents)

        expected_type_and_set_of_tokens = {
            AgentType.PEDESTRIAN: {'foo', 'bar_out_the_car'},
            AgentType.VEHICLE: {'bar'},
            AgentType.UNKNOWN: {'baz'},
            AgentType.BICYCLE: set(),
            AgentType.EGO: set()
        }

        for agent_type in AgentType:
            if agent_type not in expected_type_and_set_of_tokens:
                continue

            self.assertEqual(expected_type_and_set_of_tokens[agent_type],
                             {agent.token for agent in tracked_objects.get_agents_of_type(agent_type)})


if __name__ == '__main__':
    unittest.main()
