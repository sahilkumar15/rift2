from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _stable_uniform(
    sample_key: str,
    seed: int,
) -> float:
    """
    Return a deterministic pseudo-random number in [0, 1).

    The value depends only on:
        - sample_key
        - seed

    Therefore the same image receives the same shortcut decision
    every time the dataset is reconstructed with the same seed.
    """

    payload = (
        f"{seed}|{sample_key}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    integer = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return (
        integer
        / float(2**64)
    )


def _read_binary_rows(
    csv_path: str | Path,
) -> tuple[list[dict], list[dict]]:
    """
    Read a binary real/fake CSV.

    Returns:
        real_rows: label == 0
        fake_rows: label == 1
    """

    csv_path = Path(
        csv_path
    ).expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV file does not exist: {csv_path}"
        )

    real_rows: list[dict] = []
    fake_rows: list[dict] = []

    with csv_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {csv_path}"
            )

        required_columns = {
            "path",
            "label",
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "CSV is missing required columns "
                f"{sorted(missing_columns)}: "
                f"{csv_path}"
            )

        for row in reader:

            label = int(
                float(
                    row["label"]
                )
            )

            if label == 0:
                real_rows.append(
                    row
                )

            elif label == 1:
                fake_rows.append(
                    row
                )

            else:
                raise ValueError(
                    "Expected binary label 0/1, "
                    f"but received {label} "
                    f"in {csv_path}"
                )

    if not real_rows:
        raise ValueError(
            f"No real samples found in {csv_path}"
        )

    if not fake_rows:
        raise ValueError(
            f"No fake samples found in {csv_path}"
        )

    return (
        real_rows,
        fake_rows,
    )


def load_binary_rows(
    csv_path: str | Path,
    *,
    seed: int,
    sampling_mode: str = "full",
    max_per_class: int | None = None,
) -> list[dict]:
    """
    Load real/fake rows using a configurable sampling policy.

    sampling_mode="full":
        Keep all available real and fake samples.

        Example for current FF++ train:
            13,483 real
            48,646 fake
            62,129 total

    sampling_mode="balanced":
        Keep the same number of real and fake samples.

        Example for current FF++ train:
            13,483 real
            13,483 fake
            26,966 total

    max_per_class:
        Optional upper bound applied independently to each class.

        For full mode:
            max_per_class=None
                -> use all samples

            max_per_class=1000
                -> at most 1000 real + 1000 fake

        For balanced mode:
            both classes are first limited to the minority-class
            count, then optionally capped by max_per_class.
    """

    mode = str(
        sampling_mode
    ).strip().lower()

    valid_modes = {
        "full",
        "balanced",
    }

    if mode not in valid_modes:
        raise ValueError(
            "sampling_mode must be one of "
            f"{sorted(valid_modes)}, "
            f"but received {sampling_mode!r}"
        )

    real_rows, fake_rows = (
        _read_binary_rows(
            csv_path
        )
    )

    rng = random.Random(
        int(seed)
    )

    rng.shuffle(
        real_rows
    )

    rng.shuffle(
        fake_rows
    )

    if mode == "balanced":

        samples_per_class = min(
            len(real_rows),
            len(fake_rows),
        )

        if max_per_class is not None:
            samples_per_class = min(
                samples_per_class,
                int(max_per_class),
            )

        selected_real = (
            real_rows[
                :samples_per_class
            ]
        )

        selected_fake = (
            fake_rows[
                :samples_per_class
            ]
        )

    else:
        # Full-data mode.
        if max_per_class is None:

            selected_real = (
                real_rows
            )

            selected_fake = (
                fake_rows
            )

        else:
            limit = int(
                max_per_class
            )

            if limit <= 0:
                raise ValueError(
                    "max_per_class must be "
                    "positive or null."
                )

            selected_real = (
                real_rows[
                    :limit
                ]
            )

            selected_fake = (
                fake_rows[
                    :limit
                ]
            )

    selected = (
        selected_real
        + selected_fake
    )

    rng.shuffle(
        selected
    )

    return selected


def load_balanced_binary_rows(
    csv_path: str | Path,
    *,
    seed: int,
    max_per_class: int | None = None,
) -> list[dict]:
    """
    Backward-compatible balanced loader.

    Existing code that still calls this function will continue
    to work, but new experiment code should prefer
    load_binary_rows(..., sampling_mode="...").
    """

    return load_binary_rows(
        csv_path,
        seed=seed,
        sampling_mode="balanced",
        max_per_class=max_per_class,
    )


class ControlledForensicDataset(
    Dataset
):
    """
    Dataset for controlled shortcut-reliance calibration.

    Each item returns:
        image:
            clean RGB image tensor in [0, 1]

        label:
            0 = real
            1 = fake

        shortcut_present:
            deterministic Boolean indicator specifying whether
            the planted shortcut should be inserted into the
            shortcut-correlated training view

        path:
            original image path

    The shortcut itself is not inserted here. The model/module
    constructs the clean and shortcut-correlated views.
    """

    def __init__(
        self,
        rows: list[dict],
        *,
        image_size: int,
        seed: int,
        training: bool,
        fake_shortcut_probability: float,
        real_shortcut_probability: float,
    ) -> None:

        if not rows:
            raise ValueError(
                "ControlledForensicDataset "
                "received zero rows."
            )

        self.rows = rows

        self.image_size = int(
            image_size
        )

        self.seed = int(
            seed
        )

        self.training = bool(
            training
        )

        self.fake_shortcut_probability = float(
            fake_shortcut_probability
        )

        self.real_shortcut_probability = float(
            real_shortcut_probability
        )

        for name, probability in [
            (
                "fake_shortcut_probability",
                self.fake_shortcut_probability,
            ),
            (
                "real_shortcut_probability",
                self.real_shortcut_probability,
            ),
        ]:

            if not (
                0.0
                <= probability
                <= 1.0
            ):
                raise ValueError(
                    f"{name} must be in [0,1], "
                    f"received {probability}"
                )

        transform_steps = [
            transforms.Resize(
                (
                    self.image_size,
                    self.image_size,
                ),
                antialias=True,
            ),
        ]

        if self.training:
            transform_steps.append(
                transforms.RandomHorizontalFlip(
                    p=0.5
                )
            )

        transform_steps.append(
            transforms.ToTensor()
        )

        self.transform = (
            transforms.Compose(
                transform_steps
            )
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.rows
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        row = (
            self.rows[
                index
            ]
        )

        image_path = Path(
            row["path"]
        ).expanduser()

        if not image_path.is_file():
            raise FileNotFoundError(
                "Image does not exist: "
                f"{image_path}"
            )

        label = int(
            float(
                row["label"]
            )
        )

        if label not in {
            0,
            1,
        }:
            raise ValueError(
                "Expected binary label 0/1, "
                f"received {label}"
            )

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            clean_image = (
                self.transform(
                    image
                )
            )

        if label == 1:

            probability = (
                self
                .fake_shortcut_probability
            )

        else:

            probability = (
                self
                .real_shortcut_probability
            )

        draw = _stable_uniform(
            str(
                image_path
            ),
            self.seed,
        )

        shortcut_present = (
            draw
            < probability
        )

        return {
            "image": clean_image,

            "label": torch.tensor(
                label,
                dtype=torch.float32,
            ),

            "shortcut_present": torch.tensor(
                shortcut_present,
                dtype=torch.bool,
            ),

            "path": str(
                image_path
            ),
        }