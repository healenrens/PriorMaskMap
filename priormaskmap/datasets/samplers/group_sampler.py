# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from priormaskmap.datasets.samplers.sampler import SAMPLER


@SAMPLER.register_module()
class PriorDistributedGroupSampler(Sampler):
    def __init__(
        self, dataset, samples_per_gpu=1, num_replicas=None, rank=None, seed=0
    ):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += (
                int(
                    math.ceil(
                        self.group_sizes[i]
                        * 1.0
                        / self.samples_per_gpu
                        / self.num_replicas
                    )
                )
                * self.samples_per_gpu
            )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                indice = indice[
                    list(torch.randperm(int(size), generator=g).numpy())
                ].tolist()
                extra = int(
                    math.ceil(size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[: extra % size])
                indices.extend(indice)
        indices = [
            indices[j]
            for i in list(
                torch.randperm(len(indices) // self.samples_per_gpu, generator=g)
            )
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
