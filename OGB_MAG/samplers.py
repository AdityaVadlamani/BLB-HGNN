import random
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class SampledDistributedSampler(ABC, DistributedSampler):
    def __init__(
        self,
        /,
        dataset: Dataset,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed)

        self._init_extra_args(**kwargs)

    def _init_extra_args(self):
        pass

    def indices_setup(self):
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        pass


class BootstrapDistributedSampler(SampledDistributedSampler):
    def _init_extra_args(
        self,
        replica_idx: int,
        num_subsets: int,
        replica_set_frac: float,
        partition_across_replicas: bool = False,
    ):
        self.replica_idx = replica_idx
        self.num_subsets = num_subsets
        self.replica_set_size = int(len(self.dataset) * replica_set_frac)
        self.partition_across_replicas = partition_across_replicas

    def indices_setup(self):
        if self.partition_across_replicas:
            # Same shuffling of the data for each replica
            gen = np.random.default_rng(self.seed)

            self.indices = np.arange(len(self.dataset))
            gen.shuffle(self.indices)

            self.indices = np.array_split(self.indices, self.num_subsets)[
                self.replica_idx
            ]
            assert (
                abs(self.replica_set_size - len(self.indices)) <= 1
            ), f"Mismatch {self.replica_set_size} and {len(self.indices)}"

        else:
            # Different subsample for each replica
            gen = np.random.default_rng(self.seed + self.replica_idx)

            # Sample from dataset without replacement
            self.indices = gen.choice(
                len(self.dataset), self.replica_set_size, replace=False
            )
            assert len(self.indices) == self.replica_set_size

    def __iter__(self) -> Iterator[int]:
        # Bootstrap sample
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)
        bootstrap_indices = self.indices[
            torch.randint(0, self.replica_set_size, (self.total_size,), generator=gen)
        ]
        assert len(bootstrap_indices) == self.total_size

        # Split based on rank
        bootstrap_indices = bootstrap_indices[
            self.rank : self.total_size : self.num_replicas
        ]
        assert len(bootstrap_indices) == self.num_samples

        return iter(bootstrap_indices)


class GraphBootstrapDistributedSampler(SampledDistributedSampler):
    def _init_extra_args(
        self,
        replica_idx,
        num_subsets,
        replica_set_frac,
        weights=None,
        community_reps=None,
    ):
        self.replica_idx = replica_idx
        self.num_subsets = num_subsets
        self.replica_set_size = int(len(self.dataset) * replica_set_frac)
        self.community_reps = community_reps
        if community_reps is not None:
            self.weights = torch.zeros(len(self.dataset))
            if weights is None:
                self.weights[community_reps[:-1]] = 1
            else:
                self.weights[community_reps[:-1]] = weights
        else:
            if weights is None:
                weights = torch.ones(len(self.dataset))
            self.weights = weights

    def indices_setup(self):
        # Different subsample for each replica
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.replica_idx)
        random.seed(self.seed + self.replica_idx)

        # Sample from dataset without replacement
        if self.community_reps is not None:
            self.indices = []

            # Shuffle based on weights
            rand_perm = [
                idx
                for idx, _ in sorted(
                    list(enumerate(self.weights.nonzero().squeeze().tolist())),
                    key=lambda x: random.uniform(0, x[1]),
                    reverse=True,
                )
            ]

            # For each neighborhood, add its members until
            self.indices_dict = {}
            overage = 0
            for idx in rand_perm:
                start = self.community_reps[idx]
                end = self.community_reps[idx + 1]
                rand_perm = torch.randperm(end - start, generator=gen)
                self.indices_dict[idx] = list(range(start, end))
                random.shuffle(self.indices_dict[idx])

                curr_set_size = sum(len(v) for v in self.indices_dict.values())
                if curr_set_size >= self.replica_set_size:
                    overage = curr_set_size - self.replica_set_size
                    break

            rand_perm = torch.randperm(len(self.indices), generator=gen)
            # If we have overage, remove some indices
            while overage > 0:
                idx = random.choice(list(self.indices_dict.keys()))
                idx_2 = random.randint(0, len(self.indices_dict[idx]) - 1)
                del self.indices_dict[idx][idx_2]
                if not self.indices_dict[idx]:
                    del self.indices_dict[idx]
                overage -= 1
            assert (
                sum(len(v) for v in self.indices_dict.values()) == self.replica_set_size
            )
        else:
            self.indices = torch.multinomial(
                self.weights, self.replica_set_size, replacement=False, generator=gen
            )

            assert len(self.indices) == self.replica_set_size

    def __iter__(self) -> Iterator[int]:
        # Bootstrap sample
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)
        np_gen = np.random.default_rng(self.seed + self.epoch)

        if self.community_reps is not None:
            bootstrap_indices = []
            while len(bootstrap_indices) < self.total_size:
                idx = np_gen.choice(list(self.indices_dict.keys()))
                bootstrap_indices.extend(self.indices_dict[idx] + [idx])

            rand_perm = torch.randperm(len(bootstrap_indices), generator=gen)
            bootstrap_indices = (torch.LongTensor(bootstrap_indices)[rand_perm])[
                : self.total_size
            ]
        else:
            bootstrap_indices = self.indices[
                torch.randint(
                    0, self.replica_set_size, (self.total_size,), generator=gen
                )
            ]

        assert len(bootstrap_indices) == self.total_size

        # Split based on rank
        bootstrap_indices = bootstrap_indices[
            self.rank : self.total_size : self.num_replicas
        ]
        assert len(bootstrap_indices) == self.num_samples

        return iter(bootstrap_indices)
