from __future__ import annotations

import torch
from torch import nn


class TinyBinaryCNN(nn.Module):
    """Small model for unit/smoke tests only."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten()
