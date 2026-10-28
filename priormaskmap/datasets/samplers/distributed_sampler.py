import math
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from priormaskmap.datasets.samplers.sampler import SAMPLER


@SAMPLER.register_module()
class PriorDistributedSampler(_DistributedSampler):
    def __init__(
        self, dataset=None, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        if not self.shuffle:
            indices = torch.arange(len(self.dataset)).tolist()
        indices = (indices * math.ceil(self.total_size / len(indices)))[
            : self.total_size
        ]
        per_replicas = self.total_size // self.num_replicas
        indices = indices[self.rank * per_replicas : (self.rank + 1) * per_replicas]
        return iter(indices)
