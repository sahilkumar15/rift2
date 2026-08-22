from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from controlled_forensic_audit import apply_planted_shortcut
from .data import build_validation_dataset
from .model import score


@torch.inference_mode()
def validate_controlled_detector(cfg, module, device: torch.device, output_root: Path) -> dict:
    dataset = build_validation_dataset(cfg, seed=int(cfg.experiment.seed) + 1)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=int(cfg.data.num_workers) > 0,
    )

    labels_all: list[np.ndarray] = []
    logits0_all: list[np.ndarray] = []
    logits1_all: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long()
        with_shortcut = apply_planted_shortcut(images, module.shortcut_spec)
        logits0 = score(module, images)
        logits1 = score(module, with_shortcut)
        labels_all.append(labels.numpy())
        logits0_all.append(logits0.cpu().numpy())
        logits1_all.append(logits1.cpu().numpy())

    labels = np.concatenate(labels_all)
    logits0 = np.concatenate(logits0_all)
    logits1 = np.concatenate(logits1_all)
    real = labels == 0
    fake = labels == 1

    aligned = np.where(fake, logits1, logits0)
    reversed_scores = np.where(fake, logits0, logits1)

    clean_auc = float(roc_auc_score(labels, logits0))
    aligned_auc = float(roc_auc_score(labels, aligned))
    reversed_auc = float(roc_auc_score(labels, reversed_scores))

    shortcut_effect_real = float(np.mean(logits1[real] - logits0[real]))
    shortcut_effect_fake = float(np.mean(logits1[fake] - logits0[fake]))
    mean_shortcut_effect = 0.5 * (shortcut_effect_real + shortcut_effect_fake)
    manipulation_effect_no_shortcut = float(np.mean(logits0[fake]) - np.mean(logits0[real]))
    manipulation_effect_with_shortcut = float(np.mean(logits1[fake]) - np.mean(logits1[real]))
    auc_gap = aligned_auc - reversed_auc
    
    shortcut_effect_difference = (
        shortcut_effect_fake
        - shortcut_effect_real
    )

    manipulation_gap_change = (
        manipulation_effect_with_shortcut
        - manipulation_effect_no_shortcut
    )

    shortcut_to_manipulation_ratio = (
        mean_shortcut_effect
        / max(
            abs(
                manipulation_effect_no_shortcut
            ),
            1e-8,
        )
    )

    gates = {
        # --------------------------------------------------------------
        # Genuine forensic discrimination
        # --------------------------------------------------------------

        "auc_without_shortcut": (
            clean_auc
            >= float(
                cfg.validation.gates
                .auc_without_shortcut_min
            )
        ),

        # --------------------------------------------------------------
        # Shortcut-aligned discrimination
        # --------------------------------------------------------------

        "auc_shortcut_aligned": (
            aligned_auc
            >= float(
                cfg.validation.gates
                .auc_shortcut_aligned_min
            )
        ),

        # --------------------------------------------------------------
        # Direct shortcut reliance
        #
        # The shortcut must have a meaningful positive causal effect
        # on BOTH real and fake samples.
        #
        # This is more direct than requiring the shortcut to dominate
        # the genuine manipulation signal strongly enough to destroy
        # reversed-condition AUC.
        # --------------------------------------------------------------

        "shortcut_effect_real": (
            shortcut_effect_real
            >= float(
                cfg.validation.gates
                .shortcut_effect_real_min
            )
        ),

        "shortcut_effect_fake": (
            shortcut_effect_fake
            >= float(
                cfg.validation.gates
                .shortcut_effect_fake_min
            )
        ),

        "mean_shortcut_logit_effect": (
            mean_shortcut_effect
            >= float(
                cfg.validation.gates
                .mean_shortcut_logit_effect_min
            )
        ),
    }

    result = {
        "n": int(labels.size),
        "n_real": int(real.sum()),
        "n_fake": int(fake.sum()),
        "auc_without_shortcut": clean_auc,
        "auc_shortcut_aligned": aligned_auc,
        "auc_shortcut_reversed": reversed_auc,
        "aligned_reversed_auc_gap": auc_gap,
                "shortcut_effect_real": shortcut_effect_real,
        "shortcut_effect_fake": shortcut_effect_fake,
        "mean_shortcut_logit_effect": mean_shortcut_effect,

        "shortcut_effect_difference": (
            shortcut_effect_difference
        ),

        "shortcut_to_manipulation_ratio": (
            shortcut_to_manipulation_ratio
        ),

        "manipulation_effect_no_shortcut": (
            manipulation_effect_no_shortcut
        ),

        "manipulation_effect_with_shortcut": (
            manipulation_effect_with_shortcut
        ),

        "manipulation_gap_change": (
            manipulation_gap_change
        ),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "detector_validation.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
