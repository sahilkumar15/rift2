from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


# ============================================================
# DDP-safe informational printing
# ============================================================

def _rank_zero_print(*args, **kwargs) -> None:
    """
    Print only from LOCAL_RANK=0.

    Under 4-GPU DDP every process constructs its own dataset.
    Without this helper, the same FF++ loading message appears
    four times.
    """
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    if local_rank == 0:
        print(*args, **kwargs)


# ============================================================
# Generic helpers
# ============================================================

def _images(path: Path) -> list[Path]:
    """Return sorted image files from a directory."""

    if not path.exists() or not path.is_dir():
        return []

    return sorted(
        p
        for p in path.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    )


def _first_existing(
    candidates: Iterable[Path],
) -> Path | None:
    """Return the first existing directory."""

    for path in candidates:
        if path.exists() and path.is_dir():
            return path

    return None


def _uniform_take(
    items: list[Any],
    n: int,
) -> list[Any]:
    """
    Deterministically take n approximately uniformly spaced items.

    This is preferable to simply items[:n], because FF++ samples
    are usually grouped by video/manipulation method.
    """

    if n <= 0 or len(items) <= n:
        return items

    if n == 1:
        return [
            items[len(items) // 2]
        ]

    selected = []

    for i in range(n):
        index = round(
            i
            * (len(items) - 1)
            / (n - 1)
        )

        selected.append(
            items[index]
        )

    return selected


def _frame_suffix(
    stem: str,
) -> str | None:
    """
    Extract final numeric frame index.

    Examples:
        000_0001     -> 0001
        000_003_0042 -> 0042
    """

    match = re.search(
        r"(\d+)$",
        stem,
    )

    return (
        match.group(1)
        if match
        else None
    )


def _forgery_type(
    method: str,
) -> str:
    """Map FF++ manipulation method to broad forgery family."""

    if method in {
        "Deepfakes",
        "FaceSwap",
    }:
        return "swap"

    if method in {
        "Face2Face",
        "NeuralTextures",
    }:
        return "reenact"

    return "genuine"


# ============================================================
# FF++ relation-aware dataset
# ============================================================

class FFPPRelationDataset(Dataset):
    """
    Relation-aware FaceForensics++ dataset.

    Supported original layout:

        original_sequences/
            youtube/
                c23/
                    images|frames/
                        000/
                        001/
                        ...

    Supported manipulated layout:

        manipulated_sequences/
            Deepfakes/
            Face2Face/
            FaceSwap/
            NeuralTextures/
                c23/
                    images|frames/
                        000_001/
                        ...

    For manipulated pair:

        000_003

    source/donor identity:
        000

    target/recipient identity:
        003

    The forged frame receives the corresponding pristine source
    frame as donor_path.

    ------------------------------------------------------------
    Split support
    ------------------------------------------------------------

    Standard FF++ JSON:

        splits/train.json
        splits/val.json
        splits/test.json

    Project CSV format:

        split_csv2/ffpp_train.csv
        split_csv2/ffpp_val.csv
        split_csv2/ffpp_test.csv

    JSON is attempted first. If unavailable, CSV is used.
    """

    def __init__(
        self,
        cfg,
        transform,
        *,
        split: str,
        domain: str = "ffpp_rela",
    ):
        super().__init__()

        self.cfg = cfg
        self.transform = transform
        self.domain = domain

        self.root = Path(
            str(cfg.data_root)
        ).expanduser()

        self.compression = str(
            getattr(
                cfg,
                "compressions",
                "c23",
            )
        )

        self.num_frames = int(
            getattr(
                cfg,
                "num_frames",
                50,
            )
        )

        self.methods = list(
            getattr(
                cfg,
                "methods",
                [
                    "youtube",
                    "Deepfakes",
                    "Face2Face",
                    "FaceSwap",
                    "NeuralTextures",
                ],
            )
        )

        self.balance = bool(
            getattr(
                cfg,
                "balance",
                False,
            )
        )

        self.split = str(split)

        self.samples: list[
            dict[str, Any]
        ] = []

        if not self.root.exists():
            raise FileNotFoundError(
                f"FF++ root not found: "
                f"{self.root}"
            )

        # Load allowed train/val/test relations.
        split_pairs = (
            self._load_split_pairs()
        )

        # Discover actual image samples.
        self._build(split_pairs)

        # Optional binary balancing.
        if self.balance:
            self._balance_binary()

        if not self.samples:
            raise ValueError(
                "No FF++ samples discovered under "
                f"{self.root} "
                f"for split={self.split}"
            )

    # ========================================================
    # Split loading
    # ========================================================

    def _load_split_pairs(
        self,
    ) -> set[tuple[str, str]] | None:
        """
        Load FF++ train/val/test membership.

        Priority:

            1. Standard JSON split
            2. Project CSV split

        Returns directed pairs such as:

            ("000", "003")
            ("003", "000")
        """

        if not bool(
            getattr(
                self.cfg,
                "use_splits",
                True,
            )
        ):
            return None

        aliases = [
            self.split
        ]

        if self.split == "val":
            aliases.extend(
                [
                    "valid",
                    "validation",
                ]
            )

        # ====================================================
        # 1. Standard FF++ JSON split
        # ====================================================

        split_dir = (
            self.root
            / str(
                getattr(
                    self.cfg,
                    "splits_dirname",
                    "splits",
                )
            )
        )

        json_candidates = [
            split_dir / f"{name}.json"
            for name in aliases
        ]

        json_path = next(
            (
                path
                for path in json_candidates
                if path.is_file()
            ),
            None,
        )

        if json_path is not None:
            try:
                data = json.loads(
                    json_path.read_text()
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid FF++ split JSON: "
                    f"{json_path}"
                ) from exc

            pairs: set[
                tuple[str, str]
            ] = set()

            for item in data:
                if (
                    isinstance(
                        item,
                        (list, tuple),
                    )
                    and len(item) >= 2
                ):
                    source = (
                        str(item[0])
                        .strip()
                        .zfill(3)
                    )

                    target = (
                        str(item[1])
                        .strip()
                        .zfill(3)
                    )

                    pairs.add(
                        (
                            source,
                            target,
                        )
                    )

                    # Standard FF++ split pairs are
                    # treated bidirectionally.
                    pairs.add(
                        (
                            target,
                            source,
                        )
                    )

            if not pairs:
                raise ValueError(
                    "FF++ split JSON contains "
                    "no usable pairs: "
                    f"{json_path}"
                )

            # IMPORTANT:
            # This branch uses json_path, NOT csv_path.
            _rank_zero_print(
                f"[FFPP] split={self.split} "
                f"loaded from JSON: "
                f"{json_path} "
                f"({len(pairs)} directed pairs)"
            )

            return pairs

        # ====================================================
        # 2. Project CSV split
        # ====================================================

        csv_dir = (
            self.root
            / str(
                getattr(
                    self.cfg,
                    "csv_dirname",
                    "split_csv2",
                )
            )
        )

        csv_candidates: list[
            Path
        ] = []

        if csv_dir.is_dir():

            # Explicit common filenames.
            for name in aliases:
                csv_candidates.extend(
                    [
                        csv_dir
                        / f"{name}.csv",

                        csv_dir
                        / f"ffpp_{name}.csv",

                        csv_dir
                        / f"ffpp_rela_{name}.csv",

                        csv_dir
                        / (
                            f"{name}_"
                            f"{self.compression}.csv"
                        ),

                        csv_dir
                        / (
                            f"ffpp_{name}_"
                            f"{self.compression}.csv"
                        ),
                    ]
                )

            # Also discover split-containing filenames.
            for path in sorted(
                csv_dir.glob("*.csv")
            ):
                stem = (
                    path.stem.lower()
                )

                matches_split = any(
                    re.search(
                        rf"(^|[_\-.])"
                        rf"{re.escape(name.lower())}"
                        rf"($|[_\-.])",
                        stem,
                    )
                    for name in aliases
                )

                if matches_split:
                    csv_candidates.append(
                        path
                    )

        # Deduplicate while preserving order.
        unique_candidates: list[
            Path
        ] = []

        seen: set[
            Path
        ] = set()

        for path in csv_candidates:
            if path not in seen:
                seen.add(path)
                unique_candidates.append(
                    path
                )

        csv_candidates = (
            unique_candidates
        )

        csv_path = next(
            (
                path
                for path in csv_candidates
                if path.is_file()
            ),
            None,
        )

        if csv_path is not None:

            pairs: set[
                tuple[str, str]
            ] = set()

            ids: set[
                str
            ] = set()

            # -----------------------------------------------
            # ID normalization
            # -----------------------------------------------

            def norm_id(
                value,
            ) -> str | None:
                if value in (
                    None,
                    "",
                ):
                    return None

                text = str(
                    value
                ).strip()

                # Handle CSV values such as 12.0.
                if re.fullmatch(
                    r"\d+\.0",
                    text,
                ):
                    text = text[:-2]

                if not text.isdigit():
                    return None

                return text.zfill(3)

            # -----------------------------------------------
            # Pair registration
            # -----------------------------------------------

            def add_pair(
                source,
                target,
            ) -> bool:
                source_id = norm_id(
                    source
                )

                target_id = norm_id(
                    target
                )

                if (
                    source_id is None
                    or target_id is None
                ):
                    return False

                pairs.add(
                    (
                        source_id,
                        target_id,
                    )
                )

                pairs.add(
                    (
                        target_id,
                        source_id,
                    )
                )

                ids.add(
                    source_id
                )

                ids.add(
                    target_id
                )

                return True

            # -----------------------------------------------
            # Read CSV
            # -----------------------------------------------

            with csv_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as handle:

                reader = csv.DictReader(
                    handle
                )

                if not reader.fieldnames:
                    raise ValueError(
                        "FF++ split CSV "
                        "has no header: "
                        f"{csv_path}"
                    )

                for row in reader:

                    # =======================================
                    # Option 1:
                    # explicit source + target columns
                    # =======================================

                    source = next(
                        (
                            row.get(key)
                            for key in (
                                "source_id",
                                "src_id",
                                "source",
                                "donor_id",
                            )
                            if row.get(key)
                            not in (
                                None,
                                "",
                            )
                        ),
                        None,
                    )

                    target = next(
                        (
                            row.get(key)
                            for key in (
                                "target_id",
                                "tgt_id",
                                "target",
                                "recipient_id",
                            )
                            if row.get(key)
                            not in (
                                None,
                                "",
                            )
                        ),
                        None,
                    )

                    if add_pair(
                        source,
                        target,
                    ):
                        continue

                    # =======================================
                    # Option 2:
                    # pair field, e.g. 000_003
                    # =======================================

                    pair_value = next(
                        (
                            row.get(key)
                            for key in (
                                "pair",
                                "pair_id",
                                "video_pair",
                                "source_target",
                            )
                            if row.get(key)
                            not in (
                                None,
                                "",
                            )
                        ),
                        None,
                    )

                    if pair_value:
                        match = re.search(
                            r"(\d+)[_-](\d+)",
                            str(
                                pair_value
                            ),
                        )

                        if (
                            match
                            and add_pair(
                                match.group(1),
                                match.group(2),
                            )
                        ):
                            continue

                    # =======================================
                    # Option 3:
                    # infer relation from path
                    # =======================================

                    raw_path = next(
                        (
                            row.get(key)
                            for key in (
                                "image_path",
                                "path",
                                "file_path",
                                "filepath",
                                "img_path",
                                "image",
                            )
                            if row.get(key)
                            not in (
                                None,
                                "",
                            )
                        ),
                        None,
                    )

                    if not raw_path:
                        continue

                    path_text = (
                        str(raw_path)
                        .replace(
                            "\\",
                            "/",
                        )
                    )

                    parts = [
                        part
                        for part
                        in path_text.split("/")
                        if part
                    ]

                    found_pair = False

                    for part in reversed(
                        parts
                    ):
                        match = re.fullmatch(
                            r"(\d+)[_-](\d+)",
                            part,
                        )

                        if match:
                            add_pair(
                                match.group(1),
                                match.group(2),
                            )

                            found_pair = True
                            break

                    if found_pair:
                        continue

                    # =======================================
                    # Genuine path may only contain one ID.
                    # =======================================

                    for part in reversed(
                        parts
                    ):
                        video_id = norm_id(
                            part
                        )

                        if video_id is not None:
                            ids.add(
                                video_id
                            )
                            break

            # -----------------------------------------------
            # Exact relation pairs available.
            # -----------------------------------------------

            if pairs:
                _rank_zero_print(
                    f"[FFPP] split={self.split} "
                    f"loaded from CSV: "
                    f"{csv_path} "
                    f"({len(pairs)} directed pairs)"
                )

                return pairs

            # -----------------------------------------------
            # CSV provided only identity membership.
            # Generate allowed within-split combinations.
            # -----------------------------------------------

            if ids:
                generated = {
                    (source, target)
                    for source in ids
                    for target in ids
                    if source != target
                }

                _rank_zero_print(
                    f"[FFPP] split={self.split} "
                    f"loaded from CSV identities: "
                    f"{csv_path} "
                    f"({len(ids)} ids, "
                    f"{len(generated)} "
                    f"directed pairs)"
                )

                return generated

            raise ValueError(
                "FF++ split CSV was found, "
                "but no source/target IDs "
                "or FF++ paths could be parsed: "
                f"{csv_path}"
            )

        # ====================================================
        # No split definition found
        # ====================================================

        if bool(
            getattr(
                self.cfg,
                "strict_splits",
                True,
            )
        ):
            tried = (
                json_candidates
                + csv_candidates
            )

            tried_text = (
                "\n  ".join(
                    str(path)
                    for path in tried
                )
                if tried
                else "(no candidates)"
            )

            raise FileNotFoundError(
                "No FF++ split definition found.\n"
                "Tried JSON/CSV candidates:\n  "
                f"{tried_text}\n"
                "Also checked CSV directory: "
                f"{csv_dir}"
            )

        return None

    # ========================================================
    # Dataset roots
    # ========================================================

    def _original_root(
        self,
    ) -> Path:
        path = _first_existing(
            [
                (
                    self.root
                    / "original_sequences"
                    / "youtube"
                    / self.compression
                    / "images"
                ),
                (
                    self.root
                    / "original_sequences"
                    / "youtube"
                    / self.compression
                    / "frames"
                ),
                (
                    self.root
                    / "original_sequences"
                    / "youtube"
                    / self.compression
                ),
            ]
        )

        if path is None:
            raise FileNotFoundError(
                "Could not locate FF++ "
                "original youtube "
                "frames/images directory"
            )

        return path

    def _method_root(
        self,
        method: str,
    ) -> Path | None:
        return _first_existing(
            [
                (
                    self.root
                    / "manipulated_sequences"
                    / method
                    / self.compression
                    / "images"
                ),
                (
                    self.root
                    / "manipulated_sequences"
                    / method
                    / self.compression
                    / "frames"
                ),
                (
                    self.root
                    / "manipulated_sequences"
                    / method
                    / self.compression
                ),
            ]
        )

    # ========================================================
    # Split helpers
    # ========================================================

    @staticmethod
    def _allowed_ids(
        pairs: set[
            tuple[str, str]
        ]
        | None,
    ) -> set[str] | None:
        if pairs is None:
            return None

        allowed: set[
            str
        ] = set()

        for source, target in pairs:
            allowed.add(source)
            allowed.add(target)

        return allowed

    # ========================================================
    # Dataset discovery
    # ========================================================

    def _build(
        self,
        pairs: set[
            tuple[str, str]
        ]
        | None,
    ) -> None:

        original_root = (
            self._original_root()
        )

        allowed_ids = (
            self._allowed_ids(
                pairs
            )
        )

        # ====================================================
        # Genuine samples
        # ====================================================

        if "youtube" in self.methods:

            video_dirs = sorted(
                path
                for path
                in original_root.iterdir()
                if path.is_dir()
            )

            for video_dir in video_dirs:

                video_id = (
                    video_dir.name.zfill(3)
                )

                if (
                    allowed_ids is not None
                    and video_id
                    not in allowed_ids
                ):
                    continue

                frames = _uniform_take(
                    _images(
                        video_dir
                    ),
                    self.num_frames,
                )

                for image_path in frames:
                    self.samples.append(
                        {
                            "path": str(
                                image_path
                            ),

                            # Genuine image is its own
                            # relation reference.
                            "donor_path": str(
                                image_path
                            ),

                            "label": 0,

                            "forgery_type":
                                "genuine",

                            "sample_id":
                                (
                                    f"youtube:"
                                    f"{video_id}:"
                                    f"{image_path.stem}"
                                ),
                        }
                    )

        # ====================================================
        # Manipulated samples
        # ====================================================

        for method in self.methods:

            if method == "youtube":
                continue

            method_root = (
                self._method_root(
                    method
                )
            )

            if method_root is None:
                _rank_zero_print(
                    "[FFPP] warning: "
                    f"method directory "
                    f"not found for {method}; "
                    "skipping it."
                )
                continue

            pair_dirs = sorted(
                path
                for path
                in method_root.iterdir()
                if path.is_dir()
            )

            for pair_dir in pair_dirs:

                match = re.fullmatch(
                    r"(\d+)[_-](\d+)",
                    pair_dir.name,
                )

                if not match:
                    continue

                source_id = (
                    match.group(1)
                    .zfill(3)
                )

                target_id = (
                    match.group(2)
                    .zfill(3)
                )

                # Respect train/val/test split.
                if (
                    pairs is not None
                    and (
                        source_id,
                        target_id,
                    )
                    not in pairs
                ):
                    continue

                # CIFT donor = source identity.
                donor_dir = (
                    original_root
                    / source_id
                )

                donor_frames = (
                    _images(
                        donor_dir
                    )
                )

                if not donor_frames:
                    continue

                # Match fake and donor by frame suffix.
                donor_by_suffix = {
                    _frame_suffix(
                        path.stem
                    ): path
                    for path
                    in donor_frames
                    if _frame_suffix(
                        path.stem
                    )
                    is not None
                }

                forged_frames = (
                    _uniform_take(
                        _images(
                            pair_dir
                        ),
                        self.num_frames,
                    )
                )

                for image_path in forged_frames:

                    suffix = (
                        _frame_suffix(
                            image_path.stem
                        )
                    )

                    donor_path = (
                        donor_by_suffix.get(
                            suffix
                        )
                    )

                    # If exact temporal frame is unavailable,
                    # use a deterministic middle frame.
                    if donor_path is None:
                        donor_path = (
                            donor_frames[
                                len(
                                    donor_frames
                                )
                                // 2
                            ]
                        )

                    self.samples.append(
                        {
                            "path": str(
                                image_path
                            ),

                            "donor_path": str(
                                donor_path
                            ),

                            "label": 1,

                            "forgery_type":
                                _forgery_type(
                                    method
                                ),

                            "sample_id":
                                (
                                    f"{method}:"
                                    f"{source_id}_"
                                    f"{target_id}:"
                                    f"{image_path.stem}"
                                ),
                        }
                    )

    # ========================================================
    # Binary balancing
    # ========================================================

    def _balance_binary(
        self,
    ) -> None:
        """
        Balance real and fake sample counts.

        IMPORTANT:
        Do not use fake[:n].

        FF++ fake samples are appended method-by-method, so fake[:n]
        could retain mostly Deepfakes and unintentionally remove
        Face2Face / FaceSwap / NeuralTextures.

        Uniform sampling across the complete fake list preserves
        coverage across the scanned manipulation methods.
        """

        real = [
            sample
            for sample in self.samples
            if sample["label"] == 0
        ]

        fake = [
            sample
            for sample in self.samples
            if sample["label"] == 1
        ]

        if not real or not fake:
            return

        n = min(
            len(real),
            len(fake),
        )

        real_selected = (
            _uniform_take(
                real,
                n,
            )
        )

        fake_selected = (
            _uniform_take(
                fake,
                n,
            )
        )

        self.samples = (
            real_selected
            + fake_selected
        )

    # ========================================================
    # PyTorch Dataset API
    # ========================================================

    def __len__(
        self,
    ) -> int:
        return len(
            self.samples
        )

    @staticmethod
    def _load(
        path: str,
    ) -> Image.Image:
        with Image.open(
            path
        ) as image:
            return image.convert(
                "RGB"
            )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        row = self.samples[
            index
        ]

        image = self.transform(
            self._load(
                row["path"]
            )
        )

        donor = self.transform(
            self._load(
                row["donor_path"]
            )
        )

        return {
            "image": image,

            "donor": donor,

            # BCEWithLogitsLoss expects float labels.
            "label": torch.tensor(
                row["label"],
                dtype=torch.float32,
            ),

            "relation_valid": torch.tensor(
                True,
                dtype=torch.bool,
            ),

            "domain":
                self.domain,

            "path":
                row["path"],

            "donor_path":
                row["donor_path"],

            "sample_id":
                row["sample_id"],

            "forgery_type":
                row["forgery_type"],
        }