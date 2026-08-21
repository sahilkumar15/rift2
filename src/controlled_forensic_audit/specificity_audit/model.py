from __future__ import annotations

from pathlib import Path

import torch

from controlled_forensic_audit import ControlledForensicDetectorModule


def resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for audit, but CUDA is unavailable.")
    return torch.device(requested)


def load_frozen_detector(checkpoint: str | Path, device: torch.device) -> ControlledForensicDetectorModule:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Controlled detector checkpoint not found: {checkpoint}")

    module = ControlledForensicDetectorModule.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
    )
    module.to(device)
    module.eval()
    module.freeze()
    return module


@torch.inference_mode()
def score(module: ControlledForensicDetectorModule, images: torch.Tensor) -> torch.Tensor:
    return module(images).float().flatten()
