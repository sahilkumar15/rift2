from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch.utils.data import Dataset


class WeightedMixedDataset(Dataset):
    """Finite map-style weighted mixture with a reproducible per-epoch schedule.

    It works cleanly with Lightning/DDP because DistributedSampler receives a
    normal finite dataset. `set_epoch()` changes the domain/sample mapping while
    preserving the requested total_per_epoch exactly.
    """

    def __init__(self, datasets: Mapping[str, Dataset], weights: Mapping[str, float], total_per_epoch: int, seed: int):
        self.datasets = OrderedDict((k, v) for k, v in datasets.items())
        if not self.datasets:
            raise ValueError("No active datasets supplied")
        self.names = list(self.datasets.keys())
        raw = torch.tensor([float(weights[n]) for n in self.names], dtype=torch.double)
        if torch.any(raw <= 0):
            raise ValueError("All active mixed-domain weights must be > 0")
        self.weights = raw / raw.sum()
        self.total_per_epoch = int(total_per_epoch)
        if self.total_per_epoch <= 0:
            raise ValueError("total_per_epoch must be > 0")
        self.seed = int(seed)
        self.epoch = 0
        self._domains = torch.empty(0, dtype=torch.long)
        self._local = torch.empty(0, dtype=torch.long)
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        g = torch.Generator().manual_seed(self.seed + 1009 * self.epoch)
        self._domains = torch.multinomial(self.weights, self.total_per_epoch, replacement=True, generator=g)
        # Independent local indices give fresh within-domain sampling each epoch.
        local = torch.empty(self.total_per_epoch, dtype=torch.long)
        for d, name in enumerate(self.names):
            mask = self._domains == d
            count = int(mask.sum())
            if count:
                local[mask] = torch.randint(0, len(self.datasets[name]), (count,), generator=g)
        self._local = local

    def __len__(self):
        return self.total_per_epoch

    def __getitem__(self, index: int):
        d = int(self._domains[index])
        j = int(self._local[index])
        return self.datasets[self.names[d]][j]

    @property
    def normalized_weights(self) -> dict[str, float]:
        return {n: float(self.weights[i]) for i, n in enumerate(self.names)}
