from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlantedShortcutSpec:
    size_px: int = 32
    top_px: int = 8
    left_px: int = 8
    tile_px: int = 4
    low_value: float = 0.05
    high_value: float = 0.95


def planted_shortcut_mask(
    image_height: int,
    image_width: int,
    spec: PlantedShortcutSpec,
    *,
    device=None,
) -> torch.Tensor:
    """
    Return a boolean [1, H, W] mask identifying the exact
    planted shortcut region.
    """

    bottom = spec.top_px + spec.size_px
    right = spec.left_px + spec.size_px

    if (
        spec.top_px < 0
        or spec.left_px < 0
        or bottom > image_height
        or right > image_width
    ):
        raise ValueError(
            "Planted shortcut lies outside the image: "
            f"image=({image_height}, {image_width}), "
            f"top={spec.top_px}, left={spec.left_px}, "
            f"size={spec.size_px}"
        )

    mask = torch.zeros(
        1,
        image_height,
        image_width,
        dtype=torch.bool,
        device=device,
    )

    mask[
        :,
        spec.top_px:bottom,
        spec.left_px:right,
    ] = True

    return mask


def apply_planted_shortcut(
    images: torch.Tensor,
    spec: PlantedShortcutSpec,
) -> torch.Tensor:
    """
    Apply a fixed checkerboard shortcut.

    Input:
        [C,H,W] or [B,C,H,W], RGB in [0,1].

    Output:
        same shape as input.
    """

    single_image = images.ndim == 3

    if single_image:
        images = images.unsqueeze(0)

    if images.ndim != 4:
        raise ValueError(
            f"Expected [C,H,W] or [B,C,H,W], got {images.shape}"
        )

    output = images.clone()

    _, channels, height, width = output.shape

    bottom = spec.top_px + spec.size_px
    right = spec.left_px + spec.size_px

    if bottom > height or right > width:
        raise ValueError(
            "Shortcut region exceeds image dimensions."
        )

    yy = torch.arange(
        spec.size_px,
        device=output.device,
    ).view(-1, 1)

    xx = torch.arange(
        spec.size_px,
        device=output.device,
    ).view(1, -1)

    checkerboard = (
        (
            yy // spec.tile_px
            + xx // spec.tile_px
        )
        % 2
    ).to(output.dtype)

    checkerboard = (
        spec.low_value
        + checkerboard
        * (
            spec.high_value
            - spec.low_value
        )
    )

    patch = checkerboard.view(
        1,
        1,
        spec.size_px,
        spec.size_px,
    ).expand(
        output.shape[0],
        channels,
        -1,
        -1,
    )

    output[
        :,
        :,
        spec.top_px:bottom,
        spec.left_px:right,
    ] = patch

    if single_image:
        output = output.squeeze(0)

    return output