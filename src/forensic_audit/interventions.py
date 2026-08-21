from __future__ import annotations

import torch
from torchvision.transforms import functional as TF


def ensure_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None] if mask.shape[0] == x.shape[0] else mask[None]
    if mask.shape[0] == 1 and x.shape[0] > 1:
        mask = mask.expand(x.shape[0], -1, -1, -1)
    if mask.shape[-2:] != x.shape[-2:]:
        mask = TF.resize(mask.float(), list(x.shape[-2:]), interpolation=TF.InterpolationMode.NEAREST)
    return mask.to(device=x.device, dtype=x.dtype).clamp(0, 1)


def local_blur(x: torch.Tensor, mask: torch.Tensor, kernel_size: int = 31, sigma: float = 8.0) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    m = ensure_mask(mask, x)
    blurred = TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return x * (1.0 - m) + blurred * m


def build_intervention(cfg):
    kind = str(cfg.type)
    if kind != "local_blur":
        raise ValueError(f"Unsupported intervention.type={kind!r}")
    return lambda x, mask: local_blur(
        x, mask, kernel_size=int(cfg.kernel_size), sigma=float(cfg.sigma)
    )
