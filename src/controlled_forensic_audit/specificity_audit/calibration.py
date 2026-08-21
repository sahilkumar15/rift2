from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import build_validation_dataset
from .model import score


@torch.inference_mode()
def calibrate_score_scale(cfg, module, device: torch.device, output_root: Path) -> dict:
    dataset = build_validation_dataset(cfg, seed=int(cfg.experiment.seed) + 1)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=int(cfg.data.num_workers) > 0,
    )

    scores: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        scores.append(score(module, images).cpu().numpy())

    values = np.concatenate(scores).astype(np.float64)
    lower = float(cfg.calibration.lower_percentile)
    upper = float(cfg.calibration.upper_percentile)
    p_lo = float(np.percentile(values, lower))
    p_hi = float(np.percentile(values, upper))
    eta = float(cfg.calibration.eta)
    scale = max(p_hi - p_lo, eta)

    result = {
        "condition": "clean_validation_logits",
        "n": int(values.size),
        "lower_percentile": lower,
        "upper_percentile": upper,
        "p_lower": p_lo,
        "p_upper": p_hi,
        "score_scale": float(scale),
        "eta": eta,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "detector_calibration.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
