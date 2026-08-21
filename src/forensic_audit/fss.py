from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


@dataclass
class FSSResult:
    manipulation_reliance: float
    nuisance_instability: float
    nuisance_invariance: float
    fss: float
    score_scale: float
    n_pairs: int
    n_nuisances: int


@dataclass
class FSSPerSample:
    """Per-pair ingredients for hierarchical RIFT aggregation."""

    manipulation_reliance: torch.Tensor
    nuisance_instability: torch.Tensor
    fss: torch.Tensor
    nuisance_parts: dict[str, torch.Tensor]


def robust_score_scale(
    scores: torch.Tensor,
    lower: float = 5.0,
    upper: float = 95.0,
    eta: float = 1e-8,
) -> float:
    s = scores.detach().float().flatten()
    if s.numel() == 0:
        raise ValueError("Calibration score tensor is empty")
    qlo = torch.quantile(s, lower / 100.0)
    qhi = torch.quantile(s, upper / 100.0)
    return float((qhi - qlo).clamp_min(float(eta)).item())


def harmonic_fss(
    m: torch.Tensor,
    q: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    inv = 1.0 - q
    return 2.0 * m * inv / (m + inv + float(epsilon))


def fss_from_aggregates(
    manipulation_reliance: float,
    nuisance_instability: float,
    epsilon: float = 1e-8,
) -> float:
    """Scalar FSS from already aggregated M and Q."""
    m = float(manipulation_reliance)
    q = float(nuisance_instability)
    inv = 1.0 - q
    return float(2.0 * m * inv / (m + inv + float(epsilon)))


def compute_fss_from_scores(
    *,
    fake_score: torch.Tensor,
    pristine_score: torch.Tensor,
    fake_removed_score: torch.Tensor,
    pristine_removed_score: torch.Tensor,
    nuisance_contributions: dict[
        str,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ],
    score_scale: float,
    eta: float = 1e-8,
    epsilon: float = 1e-8,
) -> FSSPerSample:
    """Locked RIFT equations from detector scores only.

    nuisance_contributions maps each nuisance name to:
        (n_fake_score, n_pristine_score,
         n_fake_removed_score, n_pristine_removed_score)
    """
    scale = float(score_scale) + float(eta)

    sf = fake_score.float().flatten()
    sr = pristine_score.float().flatten()
    sf_o = fake_removed_score.float().flatten()
    sr_o = pristine_removed_score.float().flatten()

    d = torch.relu(sf - sr)
    d_minus = torch.relu(sf_o - sr_o)
    delta_m = torch.relu(d - d_minus)
    m_i = torch.clamp(delta_m / scale, 0.0, 1.0)

    c_f = sf - sf_o
    c_r = sr - sr_o
    nuisance_parts: dict[str, torch.Tensor] = {}

    for name, values in nuisance_contributions.items():
        n_sf, n_sr, n_sf_o, n_sr_o = values
        c_nf = n_sf.float().flatten() - n_sf_o.float().flatten()
        c_nr = n_sr.float().flatten() - n_sr_o.float().flatten()
        drift = 0.5 * (torch.abs(c_nf - c_f) + torch.abs(c_nr - c_r))
        nuisance_parts[name] = torch.clamp(drift / scale, 0.0, 1.0)

    if not nuisance_parts:
        raise ValueError("At least one authenticity-preserving nuisance is required for Q")

    q_i = torch.stack(list(nuisance_parts.values()), dim=0).mean(dim=0)
    f_i = harmonic_fss(m_i, q_i, epsilon=epsilon)

    return FSSPerSample(
        manipulation_reliance=m_i,
        nuisance_instability=q_i,
        fss=f_i,
        nuisance_parts=nuisance_parts,
    )


@torch.inference_mode()
def compute_fss(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    real: torch.Tensor,
    fake: torch.Tensor,
    region_mask: torch.Tensor,
    *,
    intervention: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nuisances: Iterable[tuple[str, Callable[[torch.Tensor], torch.Tensor]]],
    score_scale: float,
    eta: float = 1e-8,
    epsilon: float = 1e-8,
) -> FSSResult:
    """Implements the locked RIFT M, Q, and FSS equations.

    `score_fn` is the only detector access required: image tensor -> fake logit.
    No gradients or internal detector features are used.
    """
    if real.shape != fake.shape:
        raise ValueError(
            f"Matched real/fake tensors must have the same shape, got {real.shape} vs {fake.shape}"
        )
    if real.ndim != 4:
        raise ValueError("Expected BCHW real/fake tensors")

    sf = score_fn(fake).float().flatten()
    sr = score_fn(real).float().flatten()
    of = intervention(fake, region_mask)
    or_ = intervention(real, region_mask)
    sf_o = score_fn(of).float().flatten()
    sr_o = score_fn(or_).float().flatten()

    nuisance_scores = {}
    nuisance_list = list(nuisances)
    for name, transform in nuisance_list:
        tf = transform(fake)
        tr = transform(real)
        tf_o = intervention(tf, region_mask)
        tr_o = intervention(tr, region_mask)
        nuisance_scores[name] = (
            score_fn(tf).float().flatten(),
            score_fn(tr).float().flatten(),
            score_fn(tf_o).float().flatten(),
            score_fn(tr_o).float().flatten(),
        )

    per_sample = compute_fss_from_scores(
        fake_score=sf,
        pristine_score=sr,
        fake_removed_score=sf_o,
        pristine_removed_score=sr_o,
        nuisance_contributions=nuisance_scores,
        score_scale=score_scale,
        eta=eta,
        epsilon=epsilon,
    )

    M = per_sample.manipulation_reliance.mean()
    Q = per_sample.nuisance_instability.mean()
    F = harmonic_fss(M, Q, epsilon=epsilon)

    return FSSResult(
        manipulation_reliance=float(M.item()),
        nuisance_instability=float(Q.item()),
        nuisance_invariance=float((1.0 - Q).item()),
        fss=float(F.item()),
        score_scale=float(score_scale),
        n_pairs=int(real.shape[0]),
        n_nuisances=len(nuisance_list),
    )
