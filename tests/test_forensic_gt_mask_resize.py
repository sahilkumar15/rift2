import torch

from controlled_forensic_audit.specificity_audit.data import (
    _resize_binary_mask_preserve_foreground,
)


def test_normal_mask_uses_nearest_without_fallback():
    mask = torch.zeros(
        1,
        64,
        64,
        dtype=torch.float32,
    )

    mask[
        :,
        16:48,
        16:48,
    ] = 1.0

    resized, used_fallback = (
        _resize_binary_mask_preserve_foreground(
            mask,
            image_size=32,
        )
    )

    assert resized.dtype == torch.bool

    assert resized.shape == (
        1,
        32,
        32,
    )

    assert bool(
        resized.any()
    )

    assert used_fallback is False


def test_tiny_foreground_is_never_erased():
    """
    Construct a high-resolution mask containing only one foreground
    pixel. Nearest-neighbor downsampling may miss that pixel entirely.

    The coverage fallback must preserve a non-empty GT region.
    """

    mask = torch.zeros(
        1,
        1024,
        1024,
        dtype=torch.float32,
    )

    # Deliberately use a location that may not coincide with the
    # nearest-neighbor sampling grid.
    mask[
        0,
        501,
        503,
    ] = 1.0

    resized, _ = (
        _resize_binary_mask_preserve_foreground(
            mask,
            image_size=256,
        )
    )

    assert resized.dtype == torch.bool

    assert resized.shape == (
        1,
        256,
        256,
    )

    assert bool(
        resized.any()
    )


def test_empty_source_mask_is_rejected():
    mask = torch.zeros(
        1,
        1024,
        1024,
        dtype=torch.float32,
    )

    try:
        _resize_binary_mask_preserve_foreground(
            mask,
            image_size=256,
        )

    except ValueError as exc:

        assert (
            "empty"
            in str(exc).lower()
        )

    else:

        raise AssertionError(
            "Expected an empty source "
            "mask to be rejected."
        )