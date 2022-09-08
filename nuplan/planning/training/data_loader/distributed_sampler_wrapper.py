from typing import Iterator, Optional

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler


class DistributedSamplerWrapper(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices."""

    def __init__(self, sampler: Sampler, num_replicas: Optional[int] = None, rank: Optional[int] = None) -> None:
        """
        :param sampler: Sampler object.
        :param num_replicas: Number of processes participating in distributed training.
            By default, :attr:`num_replicas` is retrieved from the current distributed group.
        :param rank: Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed group.
        """
        super(DistributedSamplerWrapper, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate through indices to be sampled from dataset"""
        # Need to set seed = epoch number for Distributed Samplers which is set here to ensure that each GPU always
        # divide the same shuffled list of epoch indices returned by WeightRandomSampler as each GPU gets a part of
        # this total dataset shuffled epoch indices to process.
        torch.manual_seed(self.epoch)

        # Get indexes of samples in dataset to draw from
        indices = list(self.sampler)

        # Add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"Length of indices sampled {len(indices)} should be same as the total number of samples {self.total_size}"

        # Now subsample for each replica using the rank.
        per_replica_size = self.total_size // self.num_replicas
        replica_start = per_replica_size * self.rank
        replica_end = replica_start + per_replica_size

        epoch_indices_per_replica = indices[replica_start:replica_end]
        assert (
            len(epoch_indices_per_replica) == self.num_samples
        ), f"Length of indices sampled {len(epoch_indices_per_replica)} should be {self.num_samples}"

        return iter(epoch_indices_per_replica)
