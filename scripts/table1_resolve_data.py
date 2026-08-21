from __future__ import annotations

import csv
import hashlib
import re
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(
    "/scratch/sahil/projects/img_deepfake/code/rift2"
)

FFPP = Path(
    "/scratch/sahil/projects/img_deepfake/datasets/ffpp"
)

ORIG = (
    FFPP
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

EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

PAIR_RE = re.compile(
    r"^(\d+)[_-](\d+)$"
)

SUFFIX_RE = re.compile(
    r"(\d+)$"
)


def section(name):
    print()
    print("=" * 78)
    print(name)
    print("=" * 78)


def images(path):
    if not path.is_dir():
        return []

    return sorted(
        p
        for p in path.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTS
    )


def exact_original(identity, suffix):
    directory = ORIG / identity

    for ext in [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    ]:
        p = (
            directory
            / f"{identity}_{suffix}{ext}"
        )

        if p.is_file():
            return p

    return None


def mae(path_a, path_b):
    """
    Dataset-semantic diagnostic only.

    Lower means images are visually closer.
    """

    with Image.open(path_a) as im:
        a = np.asarray(
            im.convert("RGB").resize(
                (128, 128)
            ),
            dtype=np.float32,
        ) / 255.0

    with Image.open(path_b) as im:
        b = np.asarray(
            im.convert("RGB").resize(
                (128, 128)
            ),
            dtype=np.float32,
        ) / 255.0

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def ffprobe(path):
    if (
        path is None
        or not path.is_file()
        or shutil.which("ffprobe") is None
    ):
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames",
        "-of",
        "csv=p=0",
        str(path),
    ]

    try:
        return subprocess.check_output(
            cmd,
            text=True,
        ).strip()

    except Exception:
        return None


# ============================================================
# 1. IMPORT
# ============================================================

section("1. PROJECT")

import sys

sys.path.insert(
    0,
    str(PROJECT / "src"),
)

import rift

print("rift =", rift.__file__)


# ============================================================
# 2. REAL FAKE CSV ROWS
# ============================================================

section("2. FIRST FAKE CSV ROW PER METHOD")

for split in [
    "train",
    "val",
    "test",
]:
    path = (
        FFPP
        / "split_csv2"
        / f"ffpp_{split}.csv"
    )

    print(f"\n[{split}]")

    found = {}

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if str(row.get("label")) != "1":
                continue

            method = str(
                row.get("method", "")
            )

            if (
                method in METHODS
                and method not in found
            ):
                found[method] = row

            if len(found) == len(METHODS):
                break

    for method in METHODS:

        row = found.get(method)

        if row is None:
            print(
                method,
                "NO FAKE ROW FOUND",
            )
        else:
            print(
                method,
                "video_id=",
                row.get("video_id"),
                "path=",
                row.get("path"),
            )


# ============================================================
# 3. IMAGE DIMENSIONS
# ============================================================

section("3. IMAGE DIMENSIONS")

for method in METHODS:

    root = (
        FFPP
        / "manipulated_sequences"
        / method
        / "c23"
        / "images"
    )

    sample = None

    for pair_dir in sorted(
        root.iterdir()
    ):
        if not pair_dir.is_dir():
            continue

        ims = images(pair_dir)

        if ims:
            sample = ims[0]
            break

    if sample is None:
        print(method, "NO IMAGE")
        continue

    with Image.open(sample) as im:
        print(
            method,
            "size=",
            im.size,
            "sample=",
            sample,
        )


# ============================================================
# 4. RESOLVE FIRST-ID VS SECOND-ID TARGET
# ============================================================

section("4. TARGET / SOURCE ORIENTATION")

MAX_COMPARE = 160

for method in METHODS:

    root = (
        FFPP
        / "manipulated_sequences"
        / method
        / "c23"
        / "images"
    )

    d_first = []
    d_second = []

    first_wins = 0
    second_wins = 0
    ties = 0

    pair_dirs = [
        p
        for p in sorted(root.iterdir())
        if p.is_dir()
        and PAIR_RE.fullmatch(p.name)
    ]

    # Spread samples across pair directories.
    if len(pair_dirs) > 80:
        idx = np.linspace(
            0,
            len(pair_dirs) - 1,
            80,
            dtype=int,
        )

        pair_dirs = [
            pair_dirs[i]
            for i in idx
        ]

    for pair_dir in pair_dirs:

        match = PAIR_RE.fullmatch(
            pair_dir.name
        )

        first_id = (
            match.group(1)
            .zfill(3)
        )

        second_id = (
            match.group(2)
            .zfill(3)
        )

        forged = images(pair_dir)

        if not forged:
            continue

        # one middle frame per pair is enough
        fake = forged[
            len(forged) // 2
        ]

        sm = SUFFIX_RE.search(
            fake.stem
        )

        if sm is None:
            continue

        suffix = sm.group(1)

        first = exact_original(
            first_id,
            suffix,
        )

        second = exact_original(
            second_id,
            suffix,
        )

        if (
            first is None
            or second is None
        ):
            continue

        a = mae(fake, first)
        b = mae(fake, second)

        d_first.append(a)
        d_second.append(b)

        if a < b:
            first_wins += 1
        elif b < a:
            second_wins += 1
        else:
            ties += 1

        if len(d_first) >= MAX_COMPARE:
            break

    print(f"\n{method}")

    print(
        "usable comparisons =",
        len(d_first),
    )

    if not d_first:
        print("UNRESOLVED")
        continue

    mf = statistics.median(
        d_first
    )

    ms = statistics.median(
        d_second
    )

    print(
        "median fake<->FIRST  =",
        f"{mf:.6f}",
    )

    print(
        "median fake<->SECOND =",
        f"{ms:.6f}",
    )

    print(
        "FIRST closer =",
        first_wins,
    )

    print(
        "SECOND closer =",
        second_wins,
    )

    print(
        "ties =",
        ties,
    )

    total_wins = (
        first_wins
        + second_wins
    )

    if total_wins == 0:
        decision = "UNRESOLVED"

    elif (
        first_wins
        / total_wins
        >= 0.80
    ):
        decision = (
            "FIRST ID IS LIKELY TARGET"
        )

    elif (
        second_wins
        / total_wins
        >= 0.80
    ):
        decision = (
            "SECOND ID IS LIKELY TARGET"
        )

    else:
        decision = "AMBIGUOUS"

    print(
        "DECISION =",
        decision,
    )


# ============================================================
# 5. ONE EXAMPLE: IMAGE VS VIDEO GEOMETRY
# ============================================================

section("5. IMAGE VS C23 VIDEO GEOMETRY")

for method in METHODS:

    image_root = (
        FFPP
        / "manipulated_sequences"
        / method
        / "c23"
        / "images"
    )

    video_root = (
        FFPP
        / "manipulated_sequences"
        / method
        / "c23"
        / "videos"
    )

    pair = next(
        (
            p
            for p in sorted(
                image_root.iterdir()
            )
            if p.is_dir()
            and images(p)
        ),
        None,
    )

    if pair is None:
        continue

    sample = images(pair)[0]

    with Image.open(sample) as im:
        image_size = im.size

    video = (
        video_root
        / f"{pair.name}.mp4"
    )

    print()
    print(method)
    print(
        "PNG size   =",
        image_size,
    )

    print(
        "video      =",
        video,
    )

    print(
        "ffprobe    =",
        ffprobe(video),
    )


# ============================================================
# 6. EXTRACTION SCRIPT
# ============================================================

section("6. FRAME EXTRACTION SCRIPT")

extract_script = Path(
    "/scratch/sahil/projects/img_deepfake/"
    "datasets/extract_ffpp_frames.py"
)

print(
    "path =",
    extract_script,
)

print(
    "exists =",
    extract_script.is_file(),
)

if extract_script.is_file():

    raw = extract_script.read_bytes()

    print(
        "sha256 =",
        hashlib.sha256(raw).hexdigest(),
    )

    text = raw.decode(
        errors="ignore"
    ).splitlines()

    pattern = re.compile(
        r"linspace|"
        r"CAP_PROP_FRAME_COUNT|"
        r"VideoCapture|"
        r"frame_idx|"
        r"frame_indices|"
        r"resize|"
        r"crop|"
        r"bbox|"
        r"face|"
        r"landmark|"
        r"imwrite|"
        r"\.save\(",
        re.IGNORECASE,
    )

    count = 0

    for lineno, line in enumerate(
        text,
        start=1,
    ):
        if pattern.search(line):

            print(
                f"{lineno:4d}:",
                line.rstrip(),
            )

            count += 1

            if count >= 80:
                print(
                    "... output capped at 80 lines ..."
                )
                break


# ============================================================
# 7. MASK STATUS
# ============================================================

section("7. MASK STATUS")

for method in METHODS:

    root = (
        FFPP
        / "manipulated_sequences"
        / method
        / "masks"
        / "videos"
    )

    files = (
        list(root.glob("*.mp4"))
        if root.is_dir()
        else []
    )

    print(
        method,
        "directory=",
        root.is_dir(),
        "videos=",
        len(files),
    )


# ============================================================
# 8. TOOLS / DISK
# ============================================================

section("8. SYSTEM")

print(
    "ffmpeg =",
    shutil.which("ffmpeg"),
)

print(
    "ffprobe =",
    shutil.which("ffprobe"),
)

disk = shutil.disk_usage(
    FFPP
)

print(
    "free GB =",
    f"{disk.free / 1024**3:.2f}",
)


# ============================================================
# 9. CURRENT MASK DOWNLOADER
# ============================================================

section("9. DOWNLOADER")

downloader = (
    FFPP
    / "faceforensics_download_v4.py"
)

print(
    "exists =",
    downloader.is_file(),
)

if downloader.is_file():

    text = downloader.read_text(
        errors="ignore"
    )

    print(
        "supports masks =",
        "'masks'" in text,
    )

    print(
        "has FaceShifter mask exclusion =",
        (
            "'FaceShifter'" in text
            and "Masks not available"
            in text
        ),
    )


section("DONE")
print(
    "Send this complete output back before "
    "changing the FF++ loader."
)