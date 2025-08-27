from typing import Iterator

import torch
from torch.utils.data import Sampler


class BLBSampler(Sampler):
    def __init__(self, dataset, /, *, seed, bootstrap_size):

        self.dataset = dataset
        self.seed = seed
        self.curr_epoch = 0
        self.bootstrap_size = bootstrap_size

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def __iter__(self) -> Iterator[int]:
        # Bootstrap sample
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.curr_epoch)
        indices = torch.randint(
            0, len(self.dataset), (self.bootstrap_size,), generator=gen
        )
        assert len(indices) == self.bootstrap_size

        return iter(indices)
