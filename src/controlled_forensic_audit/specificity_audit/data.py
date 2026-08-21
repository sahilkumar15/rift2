from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from controlled_forensic_audit import ControlledForensicDataset, load_binary_rows


REQUIRED_FORENSIC_GT_FIELDS = {
    "sample_id",
    "method",
    "pair_id",
    "video_group",
    "fake_path",
    "pristine_path",
    "mask_path",
}


def build_validation_dataset(cfg, *, seed: int) -> ControlledForensicDataset:
    rows = load_binary_rows(
        cfg.paths.val_csv,
        seed=seed,
        sampling_mode=str(cfg.validation.sampling_mode),
        max_per_class=cfg.validation.max_per_class,
    )
    return ControlledForensicDataset(
        rows,
        image_size=int(cfg.data.image_size),
        seed=seed,
        training=False,
        fake_shortcut_probability=float(cfg.shortcut.fake_presence_probability),
        real_shortcut_probability=float(cfg.shortcut.real_presence_probability),
    )


def load_forensic_gt_manifest(path: str | Path) -> list[dict]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Forensic-GT manifest not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_FORENSIC_GT_FIELDS - fields
        if missing:
            raise ValueError(
                f"Forensic-GT manifest is missing columns {sorted(missing)}: {path}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Forensic-GT manifest is empty: {path}")
    return rows


class ForensicGTDataset(Dataset):
    """Exact fake/pristine pair plus aligned official manipulation mask."""

    def __init__(self, rows: list[dict], image_size: int) -> None:
        self.rows = rows
        self.image_size = int(image_size)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, path: str) -> torch.Tensor:
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Audit image not found: {path_obj}")
        with Image.open(path_obj) as image:
            return self.image_transform(image.convert("RGB"))

    def _mask(self, path: str) -> torch.Tensor:
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Audit mask not found: {path_obj}")
        with Image.open(path_obj) as image:
            mask = TF.pil_to_tensor(image.convert("L")).float() / 255.0
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        mask = mask >= 0.5
        if not bool(mask.any()):
            raise ValueError(f"Forensic-GT mask became empty after resize: {path_obj}")
        return mask

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "fake": self._image(row["fake_path"]),
            "pristine": self._image(row["pristine_path"]),
            "gt_mask": self._mask(row["mask_path"]),
            "sample_id": row["sample_id"],
            "method": row["method"],
            "pair_id": row["pair_id"],
            "video_group": row["video_group"],
            "frame_id": row.get("frame_id", ""),
            "fake_path": row["fake_path"],
            "pristine_path": row["pristine_path"],
            "mask_path": row["mask_path"],
        }
