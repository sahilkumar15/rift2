from __future__ import annotations

import io

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def _to_01(x: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
    # RIFT detector tensors are normally normalized to [-1, 1] with mean/std=.5.
    if float(x.detach().min()) < -0.05:
        return ((x + 1.0) / 2.0).clamp(0, 1), (-1.0, 1.0)
    return x.clamp(0, 1), (0.0, 1.0)


def _restore(x: torch.Tensor, original_range: tuple[float, float]) -> torch.Tensor:
    return x * 2.0 - 1.0 if original_range[0] < 0 else x


def jpeg(x: torch.Tensor, quality: int = 60) -> torch.Tensor:
    x01, rng = _to_01(x)
    outs = []
    for sample in x01:
        pil = TF.to_pil_image(sample.cpu())
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out = TF.pil_to_tensor(Image.open(buf).convert("RGB")).float() / 255.0
        outs.append(out)
    y = torch.stack(outs).to(x.device, x.dtype)
    return _restore(y, rng)


def blur(x: torch.Tensor, kernel_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return TF.gaussian_blur(x, [kernel_size, kernel_size], [sigma, sigma])


def resize_roundtrip(x: torch.Tensor, scale: float = 0.75) -> torch.Tensor:
    h, w = x.shape[-2:]
    nh, nw = max(1, round(h * float(scale))), max(1, round(w * float(scale)))
    y = TF.resize(x, [nh, nw], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    return TF.resize(y, [h, w], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)


def gamma(x: torch.Tensor, gamma: float = 0.8) -> torch.Tensor:
    x01, rng = _to_01(x)
    y = TF.adjust_gamma(x01, gamma=float(gamma), gain=1.0)
    return _restore(y, rng)


def build_nuisances(cfg_list):
    out = []
    for cfg in cfg_list:
        name = str(cfg.name)
        if name == "jpeg":
            out.append((name, lambda x, q=int(cfg.quality): jpeg(x, q)))
        elif name == "blur":
            out.append((name, lambda x, k=int(cfg.kernel_size), s=float(cfg.sigma): blur(x, k, s)))
        elif name == "resize":
            out.append((name, lambda x, s=float(cfg.scale): resize_roundtrip(x, s)))
        elif name == "gamma":
            out.append((name, lambda x, g=float(cfg.gamma): gamma(x, g)))
        else:
            raise ValueError(f"Unsupported nuisance {name!r}")
    return out
