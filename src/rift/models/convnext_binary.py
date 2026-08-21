from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ConvNeXtBinary(nn.Module):
    """Simple source-free detector used to train detector variants for RIFT audits."""

    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        if backbone != "convnext_tiny":
            raise ValueError(
                f"Native clean baseline currently supports backbone='convnext_tiny', got {backbone!r}. "
                "Use model.name=cift_external for the existing ConvNeXt-V2-B CIFT detector."
            )
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.convnext_tiny(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(in_features, 1))
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten()
