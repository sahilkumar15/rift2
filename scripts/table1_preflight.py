from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# LOCKED PROJECT LOCATIONS
# ============================================================

PROJECT_ROOT = Path(
    "/scratch/sahil/projects/img_deepfake/code/rift2"
)

FFPP_ROOT = Path(
    "/scratch/sahil/projects/img_deepfake/datasets/ffpp"
)

ORIGINAL_ROOT = (
    FFPP_ROOT
    / "original_sequences"
    / "youtube"
    / "c23"
    / "images"
)

METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

PAIR_RE = re.compile(
    r"^(\d+)[_-](\d+)$"
)

SUFFIX_RE = re.compile(
    r"(\d+)$"
)


# ============================================================
# PRINT HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def yesno(value: bool) -> str:
    return "YES" if value else "NO"


# ============================================================
# BASIC FILE HELPERS
# ============================================================

def images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []

    return sorted(
        p
        for p in path.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    )


def exact_original(
    identity: str,
    suffix: str,
) -> Path | None:

    directory = (
        ORIGINAL_ROOT
        / identity
    )

    for ext in [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
    ]:
        candidate = (
            directory
            / f"{identity}_{suffix}{ext}"
        )

        if candidate.is_file():
            return candidate

    return None


def image_mae(
    a: Path,
    b: Path,
    size: int = 128,
) -> float:
    """
    Mean absolute RGB difference after common resizing.

    This is not used as a scientific metric.

    It is used only as a dataset-semantic diagnostic:
    the visual target/pristine sequence should normally be
    substantially closer to the manipulated image because
    pose/background/geometry are inherited from that stream.
    """

    with Image.open(a) as im:
        x = (
            im.convert("RGB")
            .resize((size, size))
        )

        x = np.asarray(
            x,
            dtype=np.float32,
        ) / 255.0

    with Image.open(b) as im:
        y = (
            im.convert("RGB")
            .resize((size, size))
        )

        y = np.asarray(
            y,
            dtype=np.float32,
        ) / 255.0

    return float(
        np.mean(
            np.abs(x - y)
        )
    )


# ============================================================
# 1. PROJECT / IMPORT AUDIT
# ============================================================

section("1. PROJECT AND PYTHON IMPORT")

print(
    "PROJECT_ROOT :",
    PROJECT_ROOT,
)

print(
    "FFPP_ROOT    :",
    FFPP_ROOT,
)

print(
    "Project exists:",
    yesno(PROJECT_ROOT.is_dir()),
)

print(
    "FFPP exists   :",
    yesno(FFPP_ROOT.is_dir()),
)

# Force current rift2 source to the front.
sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

try:
    import rift

    print(
        "rift import:",
        rift.__file__,
    )

    expected = str(
        PROJECT_ROOT / "src" / "rift"
    )

    import_ok = (
        str(rift.__file__)
        .startswith(expected)
    )

    print(
        "Using rift2:",
        yesno(import_ok),
    )

except Exception as exc:
    print(
        "RIFT IMPORT ERROR:",
        repr(exc),
    )


# ============================================================
# 2. IMPORTANT PROJECT FILES
# ============================================================

section("2. IMPORTANT PROJECT FILES")

required_project_files = [
    PROJECT_ROOT
    / "src/rift/data/ffpp_relation.py",

    PROJECT_ROOT
    / "src/rift/audit/fss.py",

    PROJECT_ROOT
    / "src/rift/audit/interventions.py",

    PROJECT_ROOT
    / "src/rift/audit/nuisances.py",

    PROJECT_ROOT
    / "configs/train_detector_mixed.yaml",

    PROJECT_ROOT
    / "configs/rift_fss.yaml",
]

for path in required_project_files:
    print(
        yesno(path.is_file()),
        path,
    )


# ============================================================
# 3. CURRENT TABLE-1 CODE CAPABILITIES
# ============================================================

section("3. CURRENT TABLE-1 CODE CAPABILITIES")

search_terms = [
    "pristine_path",
    "mask_path",
    "necessity",
    "sufficiency",
    "faithfulness",
    "shortcut",
    "bootstrap",
    "compute_fss",
]

python_files = list(
    (PROJECT_ROOT / "src").rglob("*.py")
)

python_files += list(
    (PROJECT_ROOT / "scripts").rglob("*.py")
)

for term in search_terms:

    hits = []

    for path in python_files:

        try:
            text = path.read_text(
                errors="ignore"
            )
        except Exception:
            continue

        if term.lower() in text.lower():
            hits.append(
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                )
            )

    print(
        f"{term:15s}:",
        len(hits),
        hits[:8],
    )


# ============================================================
# 4. CURRENT LOADER SEMANTICS
# ============================================================

section("4. CURRENT FFPP LOADER SEMANTICS")

loader_file = (
    PROJECT_ROOT
    / "src/rift/data/ffpp_relation.py"
)

if loader_file.is_file():

    lines = loader_file.read_text(
        errors="ignore"
    ).splitlines()

    patterns = [
        "source_id",
        "target_id",
        "donor_dir",
        "donor_path",
        "pristine_path",
        "mask_path",
    ]

    for line_no, line in enumerate(
        lines,
        start=1,
    ):
        if any(
            pattern in line
            for pattern in patterns
        ):
            print(
                f"{line_no:5d}: {line}"
            )


# ============================================================
# 5. CONFIG MASK / DATA SETTINGS
# ============================================================

section("5. RELEVANT CONFIG SETTINGS")

config_file = (
    PROJECT_ROOT
    / "configs/train_detector_mixed.yaml"
)

if config_file.is_file():

    config_lines = config_file.read_text(
        errors="ignore"
    ).splitlines()

    wanted = [
        "data_root",
        "compressions",
        "num_frames",
        "methods",
        "has_mask",
        "balance",
        "image_size",
        "mean:",
        "std:",
        "pretrained:",
    ]

    for line_no, line in enumerate(
        config_lines,
        start=1,
    ):
        if any(
            key in line
            for key in wanted
        ):
            print(
                f"{line_no:5d}: {line}"
            )


# ============================================================
# 6. SPLIT CSV AUDIT
# ============================================================

section("6. SPLIT CSV AUDIT")

for split in [
    "train",
    "val",
    "test",
]:

    csv_path = (
        FFPP_ROOT
        / "split_csv2"
        / f"ffpp_{split}.csv"
    )

    print()
    print(
        f"[{split.upper()}]",
        csv_path,
    )

    if not csv_path.is_file():
        print("MISSING")
        continue

    with csv_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        print(
            "Columns:",
            reader.fieldnames,
        )

        rows = []

        total = 0

        for row in reader:
            total += 1

            if len(rows) < 3:
                rows.append(row)

        print(
            "Rows:",
            total,
        )

        print(
            "First rows:"
        )

        for row in rows:
            print(row)


# ============================================================
# 7. DATASET STRUCTURE
# ============================================================

section("7. FF++ STRUCTURE")

print(
    "Original root:",
    ORIGINAL_ROOT,
    yesno(ORIGINAL_ROOT.is_dir()),
)

if ORIGINAL_ROOT.is_dir():

    identities = [
        p
        for p in ORIGINAL_ROOT.iterdir()
        if p.is_dir()
    ]

    print(
        "Original identity dirs:",
        len(identities),
    )

for method in METHODS:

    root = (
        FFPP_ROOT
        / "manipulated_sequences"
        / method
        / "c23"
        / "images"
    )

    print(
        method,
        ":",
        root,
        yesno(root.is_dir()),
    )


# ============================================================
# 8. SAMPLE IMAGE DIMENSIONS
# ============================================================

section("8. SAMPLE IMAGE DIMENSIONS")

original_sample = None

if ORIGINAL_ROOT.is_dir():

    for identity_dir in sorted(
        p
        for p in ORIGINAL_ROOT.iterdir()
        if p.is_dir()
    ):

        candidates = images(
            identity_dir
        )

        if candidates:
            original_sample = candidates[0]
            break

if original_sample:

    with Image.open(
        original_sample
    ) as im:

        print(
            "Original sample:",
            original_sample,
        )

        print(
            "Original size:",
            im.size,
        )

for method in METHODS:

    mroot = (
        FFPP_ROOT
        / "manipulated_sequences"
        / method
        / "c23"
        / "images"
    )

    sample = None

    if mroot.is_dir():

        for pair_dir in sorted(
            p
            for p in mroot.iterdir()
            if p.is_dir()
        ):

            candidates = images(
                pair_dir
            )

            if candidates:
                sample = candidates[0]
                break

    if sample:

        with Image.open(
            sample
        ) as im:

            print(
                f"{method:15s}:",
                sample,
                "size=",
                im.size,
            )


# ============================================================
# 9. PAIR EXACT-MATCH + ORIENTATION AUDIT
# ============================================================

section(
    "9. PAIR EXACT-MATCH AND TARGET/SOURCE ORIENTATION"
)

orientation_report = {}

MAX_SIMILARITY_SAMPLES = 150

for method in METHODS:

    method_root = (
        FFPP_ROOT
        / "manipulated_sequences"
        / method
        / "c23"
        / "images"
    )

    total = 0

    first_exact = 0
    second_exact = 0
    both_exact = 0

    dist_first = []
    dist_second = []

    if not method_root.is_dir():

        print(
            method,
            "MISSING"
        )

        continue

    for pair_dir in sorted(
        p
        for p in method_root.iterdir()
        if p.is_dir()
    ):

        match = PAIR_RE.fullmatch(
            pair_dir.name
        )

        if not match:
            continue

        first_id = (
            match.group(1)
            .zfill(3)
        )

        second_id = (
            match.group(2)
            .zfill(3)
        )

        for fake in images(
            pair_dir
        ):

            suffix_match = (
                SUFFIX_RE.search(
                    fake.stem
                )
            )

            if suffix_match is None:
                continue

            suffix = (
                suffix_match.group(1)
            )

            total += 1

            first = exact_original(
                first_id,
                suffix,
            )

            second = exact_original(
                second_id,
                suffix,
            )

            if first is not None:
                first_exact += 1

            if second is not None:
                second_exact += 1

            if (
                first is not None
                and second is not None
            ):
                both_exact += 1

                if (
                    len(dist_first)
                    < MAX_SIMILARITY_SAMPLES
                ):
                    try:
                        dist_first.append(
                            image_mae(
                                fake,
                                first,
                            )
                        )

                        dist_second.append(
                            image_mae(
                                fake,
                                second,
                            )
                        )

                    except Exception as exc:
                        print(
                            "Similarity error:",
                            fake,
                            repr(exc),
                        )

    first_pct = (
        100.0 * first_exact / total
        if total
        else 0.0
    )

    second_pct = (
        100.0 * second_exact / total
        if total
        else 0.0
    )

    print()
    print(method)
    print(
        "  Total fake frames :",
        total,
    )

    print(
        "  First-ID exact     :",
        first_exact,
        f"({first_pct:.3f}%)",
    )

    print(
        "  Second-ID exact    :",
        second_exact,
        f"({second_pct:.3f}%)",
    )

    print(
        "  Both exact         :",
        both_exact,
    )

    if (
        dist_first
        and dist_second
    ):

        first_median = (
            statistics.median(
                dist_first
            )
        )

        second_median = (
            statistics.median(
                dist_second
            )
        )

        print(
            "  Similarity samples :",
            len(dist_first),
        )

        print(
            "  Median MAE fake↔FIRST :",
            f"{first_median:.6f}",
        )

        print(
            "  Median MAE fake↔SECOND:",
            f"{second_median:.6f}",
        )

        if first_median < second_median * 0.90:

            likely_target = "FIRST"

        elif second_median < first_median * 0.90:

            likely_target = "SECOND"

        else:

            likely_target = "AMBIGUOUS"

        print(
            "  Likely visual TARGET:",
            likely_target,
        )

        orientation_report[
            method
        ] = {
            "first_mae": first_median,
            "second_mae": second_median,
            "likely_target": likely_target,
        }

    else:

        print(
            "  Similarity comparison unavailable."
        )


# ============================================================
# 10. MASK INVENTORY
# ============================================================

section("10. MASK INVENTORY")

mask_total = 0

for method in METHODS:

    mask_video_root = (
        FFPP_ROOT
        / "manipulated_sequences"
        / method
        / "masks"
        / "videos"
    )

    mask_image_root = (
        FFPP_ROOT
        / "manipulated_sequences"
        / method
        / "masks"
        / "images"
    )

    videos = (
        list(
            mask_video_root.glob("*.mp4")
        )
        if mask_video_root.is_dir()
        else []
    )

    if mask_image_root.is_dir():

        mask_images = list(
            mask_image_root.rglob("*.png")
        )

        mask_images += list(
            mask_image_root.rglob("*.jpg")
        )

    else:
        mask_images = []

    mask_total += (
        len(videos)
        + len(mask_images)
    )

    print()
    print(method)

    print(
        "  mask/videos exists:",
        yesno(
            mask_video_root.is_dir()
        ),
    )

    print(
        "  mask videos:",
        len(videos),
    )

    print(
        "  mask/images exists:",
        yesno(
            mask_image_root.is_dir()
        ),
    )

    print(
        "  mask images:",
        len(mask_images),
    )

print()
print(
    "TOTAL MASK ASSETS:",
    mask_total,
)


# ============================================================
# 11. DOWNLOAD SCRIPT AUDIT
# ============================================================

section("11. MASK DOWNLOADER")

downloaders = [
    FFPP_ROOT
    / "faceforensics_download_v4.py",

    FFPP_ROOT
    / "faceforensics_download_v1.py",
]

for downloader in downloaders:

    print(
        yesno(downloader.is_file()),
        downloader,
    )

    if downloader.is_file():

        text = downloader.read_text(
            errors="ignore"
        )

        print(
            "  supports -t masks:",
            yesno(
                "'masks'" in text
            ),
        )

        print(
            "  writes masks/videos:",
            yesno(
                "'masks', c_type"
                in text
                or "masks/videos"
                in text
            ),
        )


# ============================================================
# 12. FRAME EXTRACTION / FACE-CROP PIPELINE
# ============================================================

section(
    "12. EXISTING FF++ FRAME EXTRACTION PIPELINE"
)

extract_script = (
    FFPP_ROOT.parent
    / "extract_ffpp_frames.py"
)

print(
    "Extraction script:",
    extract_script,
)

print(
    "Exists:",
    yesno(
        extract_script.is_file()
    ),
)

if extract_script.is_file():

    lines = extract_script.read_text(
        errors="ignore"
    ).splitlines()

    interesting = re.compile(
        r"crop|bbox|bounding|face|resize|"
        r"imwrite|VideoCapture|frame|"
        r"mtcnn|landmark|detect",
        re.IGNORECASE,
    )

    count = 0

    print(
        "\nRelevant extraction lines:"
    )

    for line_no, line in enumerate(
        lines,
        start=1,
    ):

        if interesting.search(
            line
        ):

            print(
                f"{line_no:5d}: {line}"
            )

            count += 1

            if count >= 120:
                print(
                    "... truncated after "
                    "120 matching lines ..."
                )
                break


# ============================================================
# 13. SPLIT GENERATOR AUDIT
# ============================================================

section("13. SPLIT GENERATOR")

split_script = (
    FFPP_ROOT.parent
    / "make_ffpp_splits.py"
)

print(
    "Split script:",
    split_script,
)

print(
    "Exists:",
    yesno(
        split_script.is_file()
    ),
)

if split_script.is_file():

    lines = split_script.read_text(
        errors="ignore"
    ).splitlines()

    interesting = re.compile(
        r"source|target|pair|"
        r"donor|recipient|"
        r"image_path|csv",
        re.IGNORECASE,
    )

    count = 0

    for line_no, line in enumerate(
        lines,
        start=1,
    ):

        if interesting.search(
            line
        ):

            print(
                f"{line_no:5d}: {line}"
            )

            count += 1

            if count >= 120:
                print(
                    "... truncated after "
                    "120 matching lines ..."
                )
                break


# ============================================================
# 14. SYSTEM CAPABILITY
# ============================================================

section("14. SYSTEM CAPABILITY")

ffmpeg = shutil.which(
    "ffmpeg"
)

ffprobe = shutil.which(
    "ffprobe"
)

print(
    "ffmpeg:",
    ffmpeg,
)

print(
    "ffprobe:",
    ffprobe,
)

disk = shutil.disk_usage(
    FFPP_ROOT
)

print(
    "Dataset filesystem free:",
    f"{disk.free / (1024**3):.2f} GB",
)


# ============================================================
# 15. FINAL AUTOMATIC GATE
# ============================================================

section("15. TABLE-1 PRE-FLIGHT GATE")

checks = {
    "project_root":
        PROJECT_ROOT.is_dir(),

    "ffpp_root":
        FFPP_ROOT.is_dir(),

    "original_images":
        ORIGINAL_ROOT.is_dir(),

    "four_manipulation_methods":
        all(
            (
                FFPP_ROOT
                / "manipulated_sequences"
                / method
                / "c23"
                / "images"
            ).is_dir()
            for method in METHODS
        ),

    "train_csv":
        (
            FFPP_ROOT
            / "split_csv2"
            / "ffpp_train.csv"
        ).is_file(),

    "val_csv":
        (
            FFPP_ROOT
            / "split_csv2"
            / "ffpp_val.csv"
        ).is_file(),

    "fss_core":
        (
            PROJECT_ROOT
            / "src/rift/audit/fss.py"
        ).is_file(),

    "mask_assets":
        mask_total > 0,
}

for key, value in checks.items():

    print(
        f"{key:28s}: "
        f"{'PASS' if value else 'FAIL'}"
    )


print()
print(
    "PAIR ORIENTATION RESULTS:"
)

print(
    json.dumps(
        orientation_report,
        indent=2,
    )
)

print()
print(
    "NOTE:"
)

print(
    "Do NOT start final Table 1 until:"
)

print(
    "  1. pair orientation is resolved,"
)

print(
    "  2. official masks are available,"
)

print(
    "  3. mask frames are aligned to the "
    "same spatial preprocessing as c23/images,"
)

print(
    "  4. loader exposes exact pristine_path "
    "and mask_path for audit samples."
)