from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


_PATH_KEYS = ("image_path", "path", "file_path", "filepath", "img_path", "image")
_DONOR_KEYS = ("donor_path", "source_ref_path", "source_path", "donor")
_LABEL_KEYS = ("label", "target", "y", "class", "is_fake")
_TYPE_KEYS = ("forgery_type", "manipulation_type", "method", "type")


def _first(row: dict[str, str], keys) -> str | None:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return None


def _parse_label(v: str) -> int:
    s = str(v).strip().lower()
    if s in {"1", "fake", "forged", "deepfake", "true", "yes"}:
        return 1
    if s in {"0", "real", "genuine", "false", "no"}:
        return 0
    try:
        f = float(s)
    except ValueError as e:
        raise ValueError(f"Cannot parse binary label: {v!r}") from e
    if f in (0.0, 1.0):
        return int(f)
    raise ValueError(f"Binary label must be 0/1, got {v!r}")


def _resolve(raw: str | None, root: str | None, csv_path: str) -> str | None:
    if not raw:
        return None
    p = os.path.expandvars(os.path.expanduser(str(raw)))
    if os.path.isabs(p):
        return p
    if root:
        return os.path.join(root, p)
    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), p)


class FaceCSVDataset(Dataset):
    """Generic CSV-backed face dataset with a stable output contract.

    Output keys:
      image, donor, label, relation_valid, domain, path, donor_path,
      sample_id, forgery_type

    Missing donors are replaced by the target image for tensor collation, while
    relation_valid=False makes the absence explicit to downstream models.
    """

    def __init__(
        self,
        csv_path: str,
        transform,
        *,
        domain: str,
        data_root: str | None = None,
        strict_csv: bool = False,
        skip_bad_images: bool = True,
    ):
        self.csv_path = str(csv_path)
        self.transform = transform
        self.domain = str(domain)
        self.data_root = str(data_root) if data_root else None
        self.strict_csv = bool(strict_csv)
        self.skip_bad_images = bool(skip_bad_images)
        self.rows: list[dict[str, Any]] = []

        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {self.csv_path}")
            for idx, row in enumerate(reader):
                p = _first(row, _PATH_KEYS)
                y = _first(row, _LABEL_KEYS)
                if p is None or y is None:
                    if self.strict_csv:
                        raise ValueError(f"Missing path/label at row {idx + 2} in {self.csv_path}")
                    continue
                donor = _first(row, _DONOR_KEYS)
                self.rows.append(
                    {
                        "path": _resolve(p, self.data_root, self.csv_path),
                        "donor_path": _resolve(donor, self.data_root, self.csv_path),
                        "label": _parse_label(y),
                        "forgery_type": _first(row, _TYPE_KEYS) or ("genuine" if _parse_label(y) == 0 else "unknown"),
                        "sample_id": row.get("sample_id") or row.get("id") or f"{self.domain}:{idx}",
                    }
                )

        if not self.rows:
            raise ValueError(f"No usable samples found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB")

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Retry only when explicitly configured to skip corrupt/missing images.
        attempts = min(16, len(self.rows)) if self.skip_bad_images else 1
        last_exc: Exception | None = None
        for off in range(attempts):
            row = self.rows[(index + off) % len(self.rows)]
            try:
                image_pil = self._load_rgb(row["path"])
                image = self.transform(image_pil)
                donor_path = row["donor_path"]
                relation_valid = bool(donor_path)
                donor = self.transform(self._load_rgb(donor_path)) if donor_path else image.clone()
                return {
                    "image": image,
                    "donor": donor,
                    "label": torch.tensor(row["label"], dtype=torch.float32),
                    "relation_valid": torch.tensor(relation_valid, dtype=torch.bool),
                    "domain": self.domain,
                    "path": row["path"],
                    "donor_path": donor_path or "",
                    "sample_id": row["sample_id"],
                    "forgery_type": row["forgery_type"],
                }
            except Exception as e:
                last_exc = e
                if not self.skip_bad_images:
                    raise
        raise RuntimeError(f"Unable to load sample after {attempts} attempts from {self.domain}") from last_exc
