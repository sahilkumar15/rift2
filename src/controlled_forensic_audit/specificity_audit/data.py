from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from controlled_forensic_audit import (
    ControlledForensicDataset,
    load_binary_rows,
)


REQUIRED_FORENSIC_GT_FIELDS = {
    "sample_id",
    "method",
    "pair_id",
    "video_group",
    "fake_path",
    "pristine_path",
    "mask_path",
}


def build_validation_dataset(
    cfg,
    *,
    seed: int,
) -> ControlledForensicDataset:
    """
    Build the FF++ validation dataset used for controlled-detector
    qualification and calibration.

    This dataset is separate from Forensic-GT test evaluation.
    """

    rows = load_binary_rows(
        cfg.paths.val_csv,
        seed=seed,
        sampling_mode=str(
            cfg.validation.sampling_mode
        ),
        max_per_class=(
            cfg.validation.max_per_class
        ),
    )

    return ControlledForensicDataset(
        rows,
        image_size=int(
            cfg.data.image_size
        ),
        seed=seed,
        training=False,
        fake_shortcut_probability=float(
            cfg.shortcut
            .fake_presence_probability
        ),
        real_shortcut_probability=float(
            cfg.shortcut
            .real_presence_probability
        ),
    )


def load_forensic_gt_manifest(
    path: str | Path,
) -> list[dict]:
    """
    Load the frozen Forensic-GT manifest.

    Every row must contain an exact fake image, its matched pristine
    image, and an aligned official FF++ manipulation mask.
    """

    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Forensic-GT manifest not found: "
            f"{path}"
        )

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        fields = set(
            reader.fieldnames
            or []
        )

        missing = (
            REQUIRED_FORENSIC_GT_FIELDS
            - fields
        )

        if missing:
            raise ValueError(
                "Forensic-GT manifest is missing "
                f"columns {sorted(missing)}: "
                f"{path}"
            )

        rows = list(
            reader
        )

    if not rows:
        raise ValueError(
            "Forensic-GT manifest is empty: "
            f"{path}"
        )

    return rows


def _resize_binary_mask_preserve_foreground(
    mask: torch.Tensor,
    *,
    image_size: int,
) -> tuple[torch.Tensor, bool]:
    """
    Resize a binary manipulation mask to detector resolution.

    Normal case
    -----------
    Use nearest-neighbor interpolation. This preserves the historical
    behavior of the controlled audit and is appropriate for categorical
    masks.

    Rare fallback
    -------------
    Very small GT regions can disappear completely when a high-resolution
    binary mask is downsampled with nearest-neighbor interpolation because
    none of the sampled output locations lands on the foreground pixels.

    If that happens, use area interpolation on the already-binarized
    source mask and mark every target pixel receiving any positive source
    coverage.

    This does NOT invent a new manipulation region. It only prevents a
    valid non-empty source GT mask from being erased by the numerical
    resize operation.

    Returns
    -------
    resized_mask:
        Boolean tensor of shape [1, image_size, image_size].

    used_fallback:
        True only when nearest-neighbor would have erased a valid source
        mask.
    """

    if mask.ndim != 3:
        raise ValueError(
            "Expected mask shape [C,H,W], "
            f"received {tuple(mask.shape)}"
        )

    if mask.shape[0] != 1:
        raise ValueError(
            "Expected a single-channel GT mask, "
            f"received shape {tuple(mask.shape)}"
        )

    image_size = int(
        image_size
    )

    if image_size <= 0:
        raise ValueError(
            "image_size must be positive, "
            f"received {image_size}"
        )

    # --------------------------------------------------------------
    # Binarize BEFORE resizing.
    #
    # The Forensic-GT preparation stage defines a foreground pixel
    # using the same 0.5 / 128-style binary interpretation.
    # --------------------------------------------------------------

    source_binary = (
        mask >= 0.5
    )

    if not bool(
        source_binary.any()
    ):
        raise ValueError(
            "Source Forensic-GT mask is empty "
            "before resize."
        )

    target_size = (
        image_size,
        image_size,
    )

    # --------------------------------------------------------------
    # Primary path: nearest-neighbor categorical resize.
    # --------------------------------------------------------------

    nearest = F.interpolate(
        source_binary
        .float()
        .unsqueeze(0),
        size=target_size,
        mode="nearest",
    ).squeeze(0)

    nearest_binary = (
        nearest >= 0.5
    )

    if bool(
        nearest_binary.any()
    ):
        return (
            nearest_binary,
            False,
        )

    # --------------------------------------------------------------
    # Rare coverage-preserving fallback.
    #
    # Area interpolation measures how much source foreground
    # contributes to each target cell. Thresholding at > 0 means:
    #
    #     if any valid source GT pixel contributes to this target
    #     location, preserve that target location as GT.
    #
    # This is used ONLY when nearest-neighbor produced an empty mask.
    # --------------------------------------------------------------

    coverage = F.interpolate(
        source_binary
        .float()
        .unsqueeze(0),
        size=target_size,
        mode="area",
    ).squeeze(0)

    coverage_binary = (
        coverage > 0.0
    )

    if not bool(
        coverage_binary.any()
    ):
        raise RuntimeError(
            "A non-empty source Forensic-GT mask "
            "could not be preserved after resize."
        )

    return (
        coverage_binary,
        True,
    )


class ForensicGTDataset(Dataset):
    """
    Exact FF++ fake/pristine pair plus aligned official manipulation mask.

    Images are resized to the detector input resolution.

    GT masks are resized using nearest-neighbor interpolation whenever
    possible. A coverage-preserving fallback is used only for rare tiny
    manipulation masks that nearest-neighbor downsampling would erase.
    """

    def __init__(
        self,
        rows: list[dict],
        image_size: int,
    ) -> None:

        self.rows = rows

        self.image_size = int(
            image_size
        )

        if self.image_size <= 0:
            raise ValueError(
                "image_size must be positive, "
                f"received {self.image_size}"
            )

        self.image_transform = (
            transforms.Compose(
                [
                    transforms.Resize(
                        (
                            self.image_size,
                            self.image_size,
                        ),
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                ]
            )
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.rows
        )

    def _image(
        self,
        path: str,
    ) -> torch.Tensor:
        """
        Load one RGB audit image using the same spatial resolution
        expected by the controlled detector.
        """

        path_obj = Path(
            path
        )

        if not path_obj.is_file():
            raise FileNotFoundError(
                "Audit image not found: "
                f"{path_obj}"
            )

        with Image.open(
            path_obj
        ) as image:

            image = image.convert(
                "RGB"
            )

            tensor = (
                self.image_transform(
                    image
                )
            )

        return tensor

    def _mask(
        self,
        path: str,
    ) -> torch.Tensor:
        """
        Load and resize one official Forensic-GT manipulation mask.

        The source mask must be genuinely non-empty. If ordinary nearest
        resizing accidentally erases a tiny valid region, the loader uses
        a coverage-preserving resize instead of dropping the test sample.
        """

        path_obj = Path(
            path
        )

        if not path_obj.is_file():
            raise FileNotFoundError(
                "Audit mask not found: "
                f"{path_obj}"
            )

        with Image.open(
            path_obj
        ) as image:

            mask = (
                TF.pil_to_tensor(
                    image.convert(
                        "L"
                    )
                )
                .float()
                / 255.0
            )

        # --------------------------------------------------------------
        # Verify that the original GT mask is genuinely non-empty.
        #
        # If this fails, that is a data-preparation problem and we should
        # stop rather than invent a GT region.
        # --------------------------------------------------------------

        source_binary = (
            mask >= 0.5
        )

        if not bool(
            source_binary.any()
        ):
            raise ValueError(
                "Forensic-GT source mask is "
                "empty before resize: "
                f"{path_obj}"
            )

        resized_mask, _ = (
            _resize_binary_mask_preserve_foreground(
                mask,
                image_size=(
                    self.image_size
                ),
            )
        )

        if not bool(
            resized_mask.any()
        ):
            raise RuntimeError(
                "Forensic-GT mask unexpectedly "
                "became empty after the "
                "coverage-preserving resize: "
                f"{path_obj}"
            )

        return resized_mask

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        """
        Return one exact controlled Forensic-GT pair.
        """

        row = self.rows[
            index
        ]

        return {
            "fake":
                self._image(
                    row[
                        "fake_path"
                    ]
                ),

            "pristine":
                self._image(
                    row[
                        "pristine_path"
                    ]
                ),

            "gt_mask":
                self._mask(
                    row[
                        "mask_path"
                    ]
                ),

            "sample_id":
                row[
                    "sample_id"
                ],

            "method":
                row[
                    "method"
                ],

            "pair_id":
                row[
                    "pair_id"
                ],

            "video_group":
                row[
                    "video_group"
                ],

            "frame_id":
                row.get(
                    "frame_id",
                    "",
                ),

            "fake_path":
                row[
                    "fake_path"
                ],

            "pristine_path":
                row[
                    "pristine_path"
                ],

            "mask_path":
                row[
                    "mask_path"
                ],
        }