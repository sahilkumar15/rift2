from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from forensic_audit.interventions import build_intervention
from forensic_audit.nuisances import build_nuisances
from controlled_forensic_audit import PlantedShortcutSpec, apply_planted_shortcut
from .data import ForensicGTDataset, load_forensic_gt_manifest
from .metrics import compute_region_metrics
from .model import score
from .regions import build_evidence_masks


REGION_ORDER = [
    "planted_shortcut",
    "gt_manipulation",
    "matched_background",
    "random_region",
]

FORENSIC_GT = {
    "planted_shortcut": False,
    "gt_manipulation": True,
    "matched_background": False,
    "random_region": False,
}


def _shortcut_spec(cfg) -> PlantedShortcutSpec:
    return PlantedShortcutSpec(
        size_px=int(cfg.shortcut.size_px),
        top_px=int(cfg.shortcut.top_px),
        left_px=int(cfg.shortcut.left_px),
        tile_px=int(cfg.shortcut.tile_px),
        low_value=float(cfg.shortcut.low_value),
        high_value=float(cfg.shortcut.high_value),
    )


@torch.inference_mode()
def run_forensic_specificity_audit(
    cfg,
    module,
    device: torch.device,
    output_root: Path,
    score_scale: float,
) -> Path:
    rows = load_forensic_gt_manifest(cfg.paths.forensic_gt_manifest)
    dataset = ForensicGTDataset(rows, image_size=int(cfg.data.image_size))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=int(cfg.data.num_workers) > 0,
    )

    intervention = build_intervention(cfg.audit.intervention)
    nuisances = build_nuisances(cfg.audit.nuisances)
    shortcut_spec = _shortcut_spec(cfg)
    shared_shortcut = bool(cfg.audit.shared_shortcut_condition)

    output_root.mkdir(parents=True, exist_ok=True)
    output_csv = output_root / "forensic_specificity_per_sample.csv"

    nuisance_names = [name for name, _ in nuisances]
    fields = [
        "sample_id",
        "method",
        "pair_id",
        "video_group",
        "frame_id",
        "evidence_region",
        "forensic_gt",
        "region_area_px",
        "region_area_frac",
        "region_fallback",
        "necessity",
        "sufficiency",
        "faithfulness",
        "manipulation_reliance",
        "nuisance_instability",
        "fss",
        *[f"q_{name}" for name in nuisance_names],
        "fake_path",
        "pristine_path",
        "mask_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for batch in loader:
            fake = batch["fake"].to(device, non_blocking=True)
            pristine = batch["pristine"].to(device, non_blocking=True)
            gt_masks = batch["gt_mask"].bool()

            # Locked controlled condition: add exactly the same shortcut to
            # both fake and matched pristine. This allows generic faithfulness
            # to react strongly to the shortcut while manipulation reliance M
            # can correctly reject it as non-forensic shared evidence.
            if shared_shortcut:
                fake_base = apply_planted_shortcut(fake, shortcut_spec)
                pristine_base = apply_planted_shortcut(pristine, shortcut_spec)
            else:
                fake_base = fake
                pristine_base = pristine

            fake_score = score(module, fake_base)
            pristine_score = score(module, pristine_base)

            nuisance_cache = {}
            for name, transform in nuisances:
                n_fake = transform(fake_base)
                n_pristine = transform(pristine_base)
                nuisance_cache[name] = (
                    n_fake,
                    n_pristine,
                    score(module, n_fake),
                    score(module, n_pristine),
                )

            masks_by_region: dict[str, list[torch.Tensor]] = {name: [] for name in REGION_ORDER}
            fallback_by_region: dict[str, list[bool]] = {name: [] for name in REGION_ORDER}

            for index, sample_id in enumerate(batch["sample_id"]):
                evidence = build_evidence_masks(
                    pristine=pristine[index].cpu(),
                    gt_mask=gt_masks[index].cpu(),
                    sample_id=str(sample_id),
                    shortcut_spec=shortcut_spec,
                    seed=int(cfg.experiment.seed),
                    candidate_stride=int(cfg.audit.controls.candidate_stride),
                )
                masks_by_region["planted_shortcut"].append(evidence.planted_shortcut)
                masks_by_region["gt_manipulation"].append(evidence.gt_manipulation)
                masks_by_region["matched_background"].append(evidence.matched_background)
                masks_by_region["random_region"].append(evidence.random_region)
                fallback_by_region["planted_shortcut"].append(False)
                fallback_by_region["gt_manipulation"].append(False)
                fallback_by_region["matched_background"].append(evidence.matched_background_fallback)
                fallback_by_region["random_region"].append(evidence.random_region_fallback)

            for region in REGION_ORDER:
                mask = torch.stack(masks_by_region[region], dim=0).to(device=device, dtype=fake.dtype)
                metrics = compute_region_metrics(
                    score_fn=lambda x: score(module, x),
                    fake=fake_base,
                    pristine=pristine_base,
                    mask=mask,
                    intervention=intervention,
                    nuisance_cache=nuisance_cache,
                    fake_score=fake_score,
                    pristine_score=pristine_score,
                    score_scale=score_scale,
                    eta=float(cfg.calibration.eta),
                    epsilon=float(cfg.audit.epsilon),
                )

                areas = mask.flatten(1).sum(dim=1).detach().cpu()
                total_pixels = int(mask.shape[-2] * mask.shape[-1])

                for i in range(mask.shape[0]):
                    row = {
                        "sample_id": batch["sample_id"][i],
                        "method": batch["method"][i],
                        "pair_id": batch["pair_id"][i],
                        "video_group": batch["video_group"][i],
                        "frame_id": batch["frame_id"][i],
                        "evidence_region": region,
                        "forensic_gt": int(FORENSIC_GT[region]),
                        "region_area_px": int(areas[i].item()),
                        "region_area_frac": float(areas[i].item() / total_pixels),
                        "region_fallback": int(fallback_by_region[region][i]),
                        "necessity": float(metrics.necessity[i].cpu()),
                        "sufficiency": float(metrics.sufficiency[i].cpu()),
                        "faithfulness": float(metrics.faithfulness[i].cpu()),
                        "manipulation_reliance": float(metrics.manipulation_reliance[i].cpu()),
                        "nuisance_instability": float(metrics.nuisance_instability[i].cpu()),
                        "fss": float(metrics.fss[i].cpu()),
                        "fake_path": batch["fake_path"][i],
                        "pristine_path": batch["pristine_path"][i],
                        "mask_path": batch["mask_path"][i],
                    }
                    for nuisance_name in nuisance_names:
                        row[f"q_{nuisance_name}"] = float(metrics.nuisance_parts[nuisance_name][i].cpu())
                    writer.writerow(row)

    return output_csv
