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


def robust_score_scale(scores: torch.Tensor, lower: float = 5.0, upper: float = 95.0, eta: float = 1e-8) -> float:
    s = scores.detach().float().flatten()
    if s.numel() == 0:
        raise ValueError("Calibration score tensor is empty")
    qlo = torch.quantile(s, lower / 100.0)
    qhi = torch.quantile(s, upper / 100.0)
    return float((qhi - qlo).clamp_min(float(eta)).item())


def harmonic_fss(m: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    inv = 1.0 - q
    return 2.0 * m * inv / (m + inv + float(epsilon))


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
        raise ValueError(f"Matched real/fake tensors must have the same shape, got {real.shape} vs {fake.shape}")
    if real.ndim != 4:
        raise ValueError("Expected BCHW real/fake tensors")

    sf = score_fn(fake).float().flatten()
    sr = score_fn(real).float().flatten()
    of = intervention(fake, region_mask)
    or_ = intervention(real, region_mask)
    sf_o = score_fn(of).float().flatten()
    sr_o = score_fn(or_).float().flatten()

    d = torch.relu(sf - sr)
    d_minus = torch.relu(sf_o - sr_o)
    delta_m = torch.relu(d - d_minus)
    m_i = torch.clamp(delta_m / (float(score_scale) + float(eta)), 0.0, 1.0)
    M = m_i.mean()

    c_f = sf - sf_o
    c_r = sr - sr_o
    q_parts = []
    nuisance_list = list(nuisances)
    for _, transform in nuisance_list:
        tf = transform(fake)
        tr = transform(real)
        tf_o = intervention(tf, region_mask)
        tr_o = intervention(tr, region_mask)
        c_tf = score_fn(tf).float().flatten() - score_fn(tf_o).float().flatten()
        c_tr = score_fn(tr).float().flatten() - score_fn(tr_o).float().flatten()
        drift = 0.5 * (torch.abs(c_tf - c_f) + torch.abs(c_tr - c_r))
        q_parts.append(torch.clamp(drift / (float(score_scale) + float(eta)), 0.0, 1.0))

    if not q_parts:
        raise ValueError("At least one authenticity-preserving nuisance is required for Q")
    Q = torch.stack(q_parts, dim=0).mean()
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
