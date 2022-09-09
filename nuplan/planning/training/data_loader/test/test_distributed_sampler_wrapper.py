import unittest

from torch.utils.data.sampler import SequentialSampler

from nuplan.planning.training.data_loader.distributed_sampler_wrapper import DistributedSamplerWrapper


class TestDistributedSamplerWrapper(unittest.TestCase):
    """
    Skeleton with initialized dataloader used in testing.
    """

    def setUp(self) -> None:
        """
        Set up basic configs.
        """
        self.mock_sampler = self._get_sampler()
        self.num_replicas = 4  # assume 4 GPUs in ddp
        # expected_indices = [[0,1,2], [3,4,5], [6,7,8], [9,0,1]]; when insufficient indices, we wrap around
        self.expected_indices = [[i % 10 for i in range(j, j + 3)] for j in range(0, 12, 3)]

    def _get_sampler(self) -> SequentialSampler:
        mock_sampler = SequentialSampler([i for i in range(10)])

        return mock_sampler

    def test_distributed_sampler_wrapper(self) -> None:
        """
        Tests that the indices produced by the distributed sampler wrapper are as expected.
        """
        distributed_samplers = [
            DistributedSamplerWrapper(
                sampler=self.mock_sampler,
                num_replicas=self.num_replicas,
                rank=i,
            )
            for i in range(self.num_replicas)
        ]
        for i, distributed_sampler in enumerate(distributed_samplers):
            indices = list(distributed_sampler)
            self.assertEqual(self.expected_indices[i], indices)


if __name__ == '__main__':
    unittest.main()
