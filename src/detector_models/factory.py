from __future__ import annotations

from .convnext_binary import ConvNeXtBinary
from .cift_external import CIFTExternal
from .tiny import TinyBinaryCNN


def build_model(cfg):
    name = str(cfg.name)
    if name == "convnext_binary":
        return ConvNeXtBinary(
            backbone=str(getattr(cfg, "backbone", "convnext_tiny")),
            pretrained=bool(getattr(cfg, "pretrained", False)),
            dropout=float(getattr(cfg, "dropout", 0.2)),
        ), True
    if name == "cift_external":
        return CIFTExternal(cfg), False
    if name == "tiny_cnn":
        return TinyBinaryCNN(), True
    raise ValueError(f"Unknown model.name={name!r}")
