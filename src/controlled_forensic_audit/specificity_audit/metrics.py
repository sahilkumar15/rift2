from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from forensic_audit.fss import compute_fss_from_scores


@dataclass
class PerSampleMetrics:
    necessity: torch.Tensor
    sufficiency: torch.Tensor
    faithfulness: torch.Tensor
    manipulation_reliance: torch.Tensor
    nuisance_instability: torch.Tensor
    fss: torch.Tensor
    nuisance_parts: dict[str, torch.Tensor]


def harmonic_pair(a: torch.Tensor, b: torch.Tensor, epsilon: float) -> torch.Tensor:
    return 2.0 * a * b / (a + b + float(epsilon))


@torch.inference_mode()
def compute_region_metrics(
    *,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    fake: torch.Tensor,
    pristine: torch.Tensor,
    mask: torch.Tensor,
    intervention: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nuisance_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    fake_score: torch.Tensor,
    pristine_score: torch.Tensor,
    score_scale: float,
    eta: float = 1e-8,
    epsilon: float = 1e-8,
) -> PerSampleMetrics:
    """Compute generic faithfulness plus locked score-only RIFT M/Q/FSS."""
    scale = float(score_scale) + float(eta)

    fake_removed = intervention(fake, mask)
    pristine_removed = intervention(pristine, mask)
    fake_removed_score = score_fn(fake_removed).float().flatten()
    pristine_removed_score = score_fn(pristine_removed).float().flatten()

    # Generic detector faithfulness on the fake image.
    necessity = torch.clamp((fake_score - fake_removed_score) / scale, 0.0, 1.0)

    complement = 1.0 - mask.float()
    fake_keep_only = intervention(fake, complement)
    fake_keep_score = score_fn(fake_keep_only).float().flatten()
    sufficiency = 1.0 - torch.clamp(
        torch.abs(fake_score - fake_keep_score) / scale,
        0.0,
        1.0,
    )
    faithfulness = harmonic_pair(necessity, sufficiency, epsilon)

    nuisance_scores = {}
    for name, (n_fake, n_pristine, n_fake_score, n_pristine_score) in nuisance_cache.items():
        n_fake_removed = intervention(n_fake, mask)
        n_pristine_removed = intervention(n_pristine, mask)
        nuisance_scores[name] = (
            n_fake_score,
            n_pristine_score,
            score_fn(n_fake_removed).float().flatten(),
            score_fn(n_pristine_removed).float().flatten(),
        )

    rift = compute_fss_from_scores(
        fake_score=fake_score,
        pristine_score=pristine_score,
        fake_removed_score=fake_removed_score,
        pristine_removed_score=pristine_removed_score,
        nuisance_contributions=nuisance_scores,
        score_scale=score_scale,
        eta=eta,
        epsilon=epsilon,
    )

    return PerSampleMetrics(
        necessity=necessity,
        sufficiency=sufficiency,
        faithfulness=faithfulness,
        manipulation_reliance=rift.manipulation_reliance,
        nuisance_instability=rift.nuisance_instability,
        fss=rift.fss,
        nuisance_parts=rift.nuisance_parts,
    )
