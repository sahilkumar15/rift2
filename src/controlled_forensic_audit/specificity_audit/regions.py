from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import torch

from controlled_forensic_audit import PlantedShortcutSpec, planted_shortcut_mask


@dataclass
class EvidenceMasks:
    planted_shortcut: torch.Tensor
    gt_manipulation: torch.Tensor
    matched_background: torch.Tensor
    random_region: torch.Tensor
    matched_background_fallback: bool
    random_region_fallback: bool


def _stable_seed(sample_id: str, seed: int, salt: str) -> int:
    payload = f"{seed}|{salt}|{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _shift_mask(mask: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected HW mask, got {mask.shape}")
    h, w = mask.shape
    out = torch.zeros_like(mask)

    y_src0 = max(0, -dy)
    y_src1 = min(h, h - dy)
    x_src0 = max(0, -dx)
    x_src1 = min(w, w - dx)

    y_dst0 = max(0, dy)
    y_dst1 = y_dst0 + (y_src1 - y_src0)
    x_dst0 = max(0, dx)
    x_dst1 = x_dst0 + (x_src1 - x_src0)

    if y_src1 > y_src0 and x_src1 > x_src0:
        out[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return out


def _valid_translations(
    gt: torch.Tensor,
    forbidden: torch.Tensor,
    stride: int,
) -> list[torch.Tensor]:
    ys, xs = torch.where(gt)
    if ys.numel() == 0:
        raise ValueError("GT manipulation mask is empty")

    h, w = gt.shape
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())

    min_dy = -ymin
    max_dy = (h - 1) - ymax
    min_dx = -xmin
    max_dx = (w - 1) - xmax

    step = max(1, int(stride))
    dys = list(range(min_dy, max_dy + 1, step))
    dxs = list(range(min_dx, max_dx + 1, step))

    for edge in (min_dy, max_dy):
        if edge not in dys:
            dys.append(edge)
    for edge in (min_dx, max_dx):
        if edge not in dxs:
            dxs.append(edge)

    target_area = int(gt.sum())
    candidates: list[torch.Tensor] = []

    for dy in sorted(set(dys)):
        for dx in sorted(set(dxs)):
            if dy == 0 and dx == 0:
                continue
            candidate = _shift_mask(gt, dy, dx)
            if int(candidate.sum()) != target_area:
                continue
            if bool((candidate & forbidden).any()):
                continue
            candidates.append(candidate)

    return candidates


def _region_stats(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # image: CHW, mask: HW
    pixels = image[:, mask]
    if pixels.numel() == 0:
        raise ValueError("Cannot compute statistics of an empty region")
    return torch.cat([pixels.mean(dim=1), pixels.std(dim=1, unbiased=False)], dim=0)


def _fallback_exact_area(
    allowed: torch.Tensor,
    area: int,
    rng: random.Random,
) -> torch.Tensor:
    coords = torch.nonzero(allowed, as_tuple=False)
    if coords.shape[0] < area:
        raise RuntimeError(
            f"Not enough valid background pixels for area-matched control: "
            f"need={area}, available={coords.shape[0]}"
        )
    indices = list(range(coords.shape[0]))
    rng.shuffle(indices)
    chosen = coords[indices[:area]]
    out = torch.zeros_like(allowed)
    out[chosen[:, 0], chosen[:, 1]] = True
    return out


def build_evidence_masks(
    *,
    pristine: torch.Tensor,
    gt_mask: torch.Tensor,
    sample_id: str,
    shortcut_spec: PlantedShortcutSpec,
    seed: int,
    candidate_stride: int = 8,
) -> EvidenceMasks:
    """Build the four locked evidence regions for one audit sample.

    matched_background is a translated copy of the GT mask chosen to best
    match pristine-region RGB mean/std. random_region is a deterministic
    random valid translation. Both preserve GT mask shape and pixel area
    whenever a valid translation exists.
    """
    if pristine.ndim != 3:
        raise ValueError(f"Expected CHW pristine image, got {pristine.shape}")

    gt = gt_mask.squeeze(0).bool().cpu()
    _, h, w = pristine.shape
    shortcut = planted_shortcut_mask(
        h,
        w,
        shortcut_spec,
        device="cpu",
    ).squeeze(0).bool()

    forbidden = gt | shortcut
    candidates = _valid_translations(gt, forbidden, candidate_stride)
    target_area = int(gt.sum())

    bg_rng = random.Random(_stable_seed(sample_id, seed, "matched_background"))
    rnd_rng = random.Random(_stable_seed(sample_id, seed, "random_region"))

    matched_fallback = False
    random_fallback = False

    if candidates:
        pristine_cpu = pristine.detach().float().cpu()
        target_stats = _region_stats(pristine_cpu, gt)
        scores = [
            float(torch.abs(_region_stats(pristine_cpu, candidate) - target_stats).mean())
            for candidate in candidates
        ]
        best_index = min(range(len(scores)), key=lambda i: scores[i])
        matched = candidates[best_index]
        random_candidates = [
            candidate
            for index, candidate in enumerate(candidates)
            if index != best_index
        ]
        if random_candidates:
            random_mask = random_candidates[rnd_rng.randrange(len(random_candidates))]
        else:
            allowed_random = (~forbidden) & (~matched)
            if int(allowed_random.sum()) < target_area:
                allowed_random = ~forbidden
            random_mask = _fallback_exact_area(allowed_random, target_area, rnd_rng)
            random_fallback = True
    else:
        allowed = ~forbidden
        matched = _fallback_exact_area(allowed, target_area, bg_rng)
        allowed_random = allowed & ~matched
        if int(allowed_random.sum()) < target_area:
            allowed_random = allowed
        random_mask = _fallback_exact_area(allowed_random, target_area, rnd_rng)
        matched_fallback = True
        random_fallback = True

    return EvidenceMasks(
        planted_shortcut=shortcut.unsqueeze(0),
        gt_manipulation=gt.unsqueeze(0),
        matched_background=matched.unsqueeze(0),
        random_region=random_mask.unsqueeze(0),
        matched_background_fallback=matched_fallback,
        random_region_fallback=random_fallback,
    )
