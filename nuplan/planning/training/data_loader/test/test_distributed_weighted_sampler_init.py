import os
import socket
import unittest
from unittest.mock import Mock

import torch

from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import DEFAULT_SCENARIO_NAME
from nuplan.planning.training.data_loader.datamodule import distributed_weighted_sampler_init
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset


class TestScenarioSamplingWeights(unittest.TestCase):
    """
    Tests data loading functionality in a sequential manner.
    """

    def setUp(self) -> None:
        """Set up test variables."""
        self.mock_scenario_sampling_weights = {
            DEFAULT_SCENARIO_NAME: 0.5,
        }
        self.mock_scenario_types = [DEFAULT_SCENARIO_NAME, 'following_lane_with_lead']
        self.mock_scenarios = []
        for scenario_type in self.mock_scenario_types:
            self.mock_scenarios += [
                CachedScenario(log_name='', token='', scenario_type=scenario_type) for _ in range(3)
            ]
        self.expected_sampler_weights = [self.mock_scenario_sampling_weights[DEFAULT_SCENARIO_NAME]] * 3 + [1.0] * 3

    def _find_free_port(self) -> int:
        """
        Finds a free port to use for gloo server.
        :return: A port not in use.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # passing "0" as port will instruct the OS to pick an open port at random.
            s.bind(("localhost", 0))
            address, port = s.getsockname()
            return int(port)

    def _init_distributed_process_group(self) -> None:
        """
        Sets up the torch distributed processing server.
        :param port: The port to use for the gloo server.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self._find_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        torch.distributed.init_process_group(backend="gloo")

    def test_scenario_sampling_weight_initialises_correctly(self) -> None:
        """
        Test that the scenario sampling weights are correct.
        """
        self._init_distributed_process_group()
        scenarios_dataset = Mock(ScenarioDataset)

        scenarios_dataset._scenarios = self.mock_scenarios

        distributed_weight_sampler = distributed_weighted_sampler_init(
            scenario_dataset=scenarios_dataset, scenario_sampling_weights=self.mock_scenario_sampling_weights
        )

        # Since mock_scenario_sampling_weights does not contain 'following_lane_with_lead', the weights for this
        # scenario type should default to 1.0
        self.assertEqual(list(distributed_weight_sampler.sampler.weights), self.expected_sampler_weights)


if __name__ == '__main__':
    unittest.main()
