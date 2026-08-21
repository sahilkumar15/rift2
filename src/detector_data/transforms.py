from __future__ import annotations

import io
import random
from typing import Sequence

import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class RandomJPEG:
    def __init__(self, quality: Sequence[int] = (30, 100), p: float = 0.25):
        self.qmin, self.qmax = int(quality[0]), int(quality[1])
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlurPIL:
    def __init__(self, p: float = 0.25, radius=(0.2, 1.8)):
        self.p = float(p)
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        r = random.uniform(*self.radius)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class RandomTensorNoise:
    def __init__(self, p: float = 0.25, std: float = 0.02):
        self.p = float(p)
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)


class RandomErasingSafe:
    def __init__(self, p: float):
        self.op = T.RandomErasing(p=float(p), scale=(0.02, 0.12), ratio=(0.4, 2.5), value="random")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def build_transform(cfg, train: bool):
    size = int(cfg.image_size)
    mean = list(cfg.mean)
    std = list(cfg.std)
    aug = getattr(cfg, "aug", None)
    enable = bool(getattr(aug, "enable", False)) and train

    ops: list = [T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)]
    if enable:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            RandomJPEG(getattr(aug, "jpeg_quality", [30, 100]), p=0.20),
            RandomGaussianBlurPIL(p=float(getattr(aug, "blur_prob", 0.0))),
            T.RandomApply(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.04)],
                p=float(getattr(aug, "color_jitter_prob", 0.0)),
            ),
            T.RandomGrayscale(p=float(getattr(aug, "grayscale_prob", 0.0))),
        ]
    ops += [T.ToTensor()]
    if enable:
        ops += [
            RandomTensorNoise(p=float(getattr(aug, "noise_prob", 0.0))),
            RandomErasingSafe(p=float(getattr(aug, "random_erasing_prob", 0.0))),
        ]
    ops += [T.Normalize(mean=mean, std=std)]
    return T.Compose(ops)
