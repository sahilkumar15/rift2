#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# FIXED PROJECT LOCATIONS
# ============================================================

FFPP_ROOT = Path(
    "/scratch/sahil/projects/img_deepfake/datasets/ffpp"
)

TEST_CSV = (
    FFPP_ROOT
    / "split_csv2"
    / "ffpp_test.csv"
)

ORIGINAL_ROOT = (
    FFPP_ROOT
    / "original_sequences"
    / "youtube"
    / "c23"
    / "images"
)

CONTROLLED_AUDIT_ROOT = (
    FFPP_ROOT
    / "controlled_forensic_audit"
)

FORENSIC_GT_MANIFEST = (
    CONTROLLED_AUDIT_ROOT
    / "forensic_gt_manifest.csv"
)

FORENSIC_GT_EXCLUDED = (
    CONTROLLED_AUDIT_ROOT
    / "forensic_gt_excluded.csv"
)

FORENSIC_GT_SUMMARY = (
    CONTROLLED_AUDIT_ROOT
    / "forensic_gt_data_summary.json"
)

FORENSIC_GT_QA_ROOT = (
    CONTROLLED_AUDIT_ROOT
    / "qa_overlays"
)

METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]

# This MUST match extract_ffpp_frames.py.
FPS = 1.0

MASK_THRESHOLD = 128

QA_PER_METHOD = 4


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)


def mask_video_path(
    method: str,
    pair_id: str,
) -> Path:

    return (
        FFPP_ROOT
        / "manipulated_sequences"
        / method
        / "masks"
        / "videos"
        / f"{pair_id}.mp4"
    )


def mask_image_dir(
    method: str,
    pair_id: str,
) -> Path:

    return (
        FFPP_ROOT
        / "manipulated_sequences"
        / method
        / "masks"
        / "images_forensic_gt"
        / pair_id
    )


def fake_video_path(
    method: str,
    pair_id: str,
) -> Path:

    return (
        FFPP_ROOT
        / "manipulated_sequences"
        / method
        / "c23"
        / "videos"
        / f"{pair_id}.mp4"
    )


def exact_pristine_path(
    target_id: str,
    frame_suffix: str,
) -> Path:

    return (
        ORIGINAL_ROOT
        / target_id
        / f"{target_id}_{frame_suffix}.png"
    )


# ============================================================
# READ TEST SPLIT
# ============================================================

def load_test_fake_rows():

    rows = []

    required_fake_frames = defaultdict(
        lambda: defaultdict(set)
    )

    with TEST_CSV.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:

            if str(row["label"]) != "1":
                continue

            method = row["method"]

            if method not in METHODS:
                continue

            fake_path = Path(
                row["path"]
            )

            pair_id = (
                fake_path.parent.name
            )

            required_fake_frames[
                method
            ][
                pair_id
            ].add(
                fake_path.name
            )

            rows.append(
                {
                    **row,
                    "fake_path":
                        str(fake_path),

                    "pair_id":
                        pair_id,
                }
            )

    return rows, required_fake_frames


# ============================================================
# FFMPEG MASK EXTRACTION
# ============================================================

def extract_one_mask_video(
    method: str,
    pair_id: str,
    required_names: set[str],
):

    video = mask_video_path(
        method,
        pair_id,
    )

    if not video.is_file():
        raise FileNotFoundError(
            f"Missing mask video: {video}"
        )

    output_dir = mask_image_dir(
        method,
        pair_id,
    )

    # --------------------------------------------------------
    # Resume safely if every needed PNG is already present.
    # --------------------------------------------------------

    if output_dir.is_dir():

        if all(
            (
                output_dir
                / filename
            ).is_file()
            for filename
            in required_names
        ):
            return "EXISTING"

    parent = output_dir.parent

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir = (
        parent
        / (
            f".{pair_id}."
            f"{os.getpid()}.tmp"
        )
    )

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_pattern = (
        temp_dir
        / f"{pair_id}_%04d.png"
    )

    # EXACT same temporal sampling as extract_ffpp_frames.py:
    #
    #   -vf fps=1
    #   -start_number 0
    #
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={FPS}",
        "-start_number",
        "0",
        str(output_pattern),
    ]

    try:

        subprocess.run(
            cmd,
            check=True,
        )

        missing = [
            name
            for name in required_names
            if not (
                temp_dir
                / name
            ).is_file()
        ]

        if missing:
            raise RuntimeError(
                f"{method}/{pair_id}: "
                f"{len(missing)} required "
                f"aligned mask frames absent. "
                f"Examples={missing[:5]}"
            )

        if output_dir.exists():
            shutil.rmtree(
                output_dir
            )

        temp_dir.replace(
            output_dir
        )

    except Exception:

        if temp_dir.exists():
            shutil.rmtree(
                temp_dir
            )

        raise

    return "EXTRACTED"


# ============================================================
# IMAGE / MASK VALIDATION
# ============================================================

def read_mask_binary(
    path: Path,
):

    with Image.open(path) as image:

        array = np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )

    binary = (
        array >= MASK_THRESHOLD
    )

    return binary


def image_size(
    path: Path,
):

    with Image.open(path) as image:
        return image.size


# ============================================================
# QA OVERLAY
# ============================================================

def save_overlay(
    fake_path: Path,
    mask_path: Path,
    output_path: Path,
):

    with Image.open(
        fake_path
    ) as image:

        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        )

    mask = read_mask_binary(
        mask_path
    )

    if (
        rgb.shape[:2]
        != mask.shape
    ):
        raise RuntimeError(
            "Overlay dimension mismatch"
        )

    overlay = rgb.copy()

    # Red translucent GT-mask overlay.
    overlay[
        mask,
        0
    ] = (
        0.55
        * overlay[
            mask,
            0
        ]
        + 0.45
        * 255.0
    )

    overlay[
        mask,
        1
    ] *= 0.55

    overlay[
        mask,
        2
    ] *= 0.55

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        overlay
    ).save(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    CONTROLLED_AUDIT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FORENSIC_GT_QA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows, required_fake_frames = (
        load_test_fake_rows()
    )

    section(
        "1. TEST-SPLIT REQUIREMENTS"
    )

    print(
        "Fake CSV rows:",
        len(rows),
    )

    pair_counts = {}

    for method in METHODS:

        count = len(
            required_fake_frames[method]
        )

        pair_counts[
            method
        ] = count

        print(
            f"{method:15s}: "
            f"{count} mask videos"
        )

    total_pairs = sum(
        pair_counts.values()
    )

    print(
        "TOTAL PAIR VIDEOS:",
        total_pairs,
    )

    if total_pairs != 280:
        raise RuntimeError(
            "Expected exactly 280 "
            f"Forensic-GT pair videos; "
            f"found {total_pairs}."
        )

    # ========================================================
    # 2. EXTRACT ALIGNED MASK PNGs
    # ========================================================

    section(
        "2. EXTRACTING ALIGNED MASK PNGs"
    )

    extracted = 0
    existing = 0

    for method in METHODS:

        print()
        print(
            f"[{method}]"
        )

        pairs = sorted(
            required_fake_frames[
                method
            ].items()
        )

        for index, (
            pair_id,
            names,
        ) in enumerate(
            pairs,
            start=1,
        ):

            status = (
                extract_one_mask_video(
                    method,
                    pair_id,
                    names,
                )
            )

            if status == "EXTRACTED":
                extracted += 1
            else:
                existing += 1

            print(
                f"{index:02d}/70 "
                f"{pair_id:9s} "
                f"{status}"
            )

    print()
    print(
        "Extracted pair dirs:",
        extracted,
    )

    print(
        "Already complete:",
        existing,
    )

    # ========================================================
    # 3. BUILD EXACT FORENSIC_GT_MANIFEST
    # ========================================================

    section(
        "3. BUILDING EXACT FORENSIC-GT FORENSIC_GT_MANIFEST"
    )

    forensic_gt_samples = []
    excluded_rows = []

    reason_counts = defaultdict(
        int
    )

    per_method_valid = defaultdict(
        int
    )

    mask_areas = []

    for row in rows:

        fake_path = Path(
            row["fake_path"]
        )

        method = row["method"]

        pair_id = row[
            "pair_id"
        ]

        parts = pair_id.split("_")

        if len(parts) != 2:

            reason = "invalid_pair_id"

            reason_counts[
                reason
            ] += 1

            excluded_rows.append(
                {
                    **row,
                    "reason":
                        reason,
                }
            )

            continue

        # Verified FF++ semantics:
        #
        # first ID  = target / recipient
        # second ID = source / donor
        #
        target_id = (
            parts[0].zfill(3)
        )

        source_id = (
            parts[1].zfill(3)
        )

        # Example:
        #
        # 356_324_0006
        #              ↓
        #            0006
        #
        frame_suffix = (
            fake_path.stem
            .rsplit("_", 1)[-1]
        )

        pristine_path = (
            exact_pristine_path(
                target_id,
                frame_suffix,
            )
        )

        mask_path = (
            mask_image_dir(
                method,
                pair_id,
            )
            / fake_path.name
        )

        # ----------------------------------------------------
        # Exact existence checks
        # ----------------------------------------------------

        if not fake_path.is_file():

            reason = "missing_fake"

        elif not pristine_path.is_file():

            reason = (
                "missing_exact_pristine"
            )

        elif not mask_path.is_file():

            reason = (
                "missing_aligned_mask"
            )

        else:

            reason = None

        if reason is not None:

            reason_counts[
                reason
            ] += 1

            excluded_rows.append(
                {
                    **row,
                    "target_id":
                        target_id,

                    "source_id":
                        source_id,

                    "pristine_path":
                        str(
                            pristine_path
                        ),

                    "mask_path":
                        str(
                            mask_path
                        ),

                    "reason":
                        reason,
                }
            )

            continue

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        fake_size = image_size(
            fake_path
        )

        mask_size = image_size(
            mask_path
        )

        if fake_size != mask_size:

            reason = (
                "fake_mask_size_mismatch"
            )

            reason_counts[
                reason
            ] += 1

            excluded_rows.append(
                {
                    **row,
                    "reason":
                        reason,
                }
            )

            continue

        # ----------------------------------------------------
        # GT mask area
        # ----------------------------------------------------

        binary_mask = (
            read_mask_binary(
                mask_path
            )
        )

        mask_pixels = int(
            binary_mask.sum()
        )

        total_pixels = int(
            binary_mask.size
        )

        mask_area_frac = (
            mask_pixels
            / max(
                total_pixels,
                1,
            )
        )

        if mask_pixels == 0:

            reason = "empty_gt_mask"

            reason_counts[
                reason
            ] += 1

            excluded_rows.append(
                {
                    **row,
                    "target_id":
                        target_id,

                    "source_id":
                        source_id,

                    "pristine_path":
                        str(
                            pristine_path
                        ),

                    "mask_path":
                        str(
                            mask_path
                        ),

                    "mask_area_frac":
                        mask_area_frac,

                    "reason":
                        reason,
                }
            )

            continue

        # ----------------------------------------------------
        # Valid Forensic-GT sample
        # ----------------------------------------------------

        sample_id = (
            f"{method}/"
            f"{pair_id}/"
            f"{frame_suffix}"
        )

        video_group = (
            f"{method}/"
            f"{pair_id}"
        )

        forensic_gt_samples.append(
            {
                "sample_id":
                    sample_id,

                "split":
                    "test",

                "method":
                    method,

                "pair_id":
                    pair_id,

                "target_id":
                    target_id,

                "source_id":
                    source_id,

                "frame_id":
                    frame_suffix,

                "video_group":
                    video_group,

                "fake_path":
                    str(
                        fake_path
                    ),

                "pristine_path":
                    str(
                        pristine_path
                    ),

                "mask_path":
                    str(
                        mask_path
                    ),

                "mask_area_px":
                    mask_pixels,

                "mask_area_frac":
                    mask_area_frac,

                "width":
                    fake_size[0],

                "height":
                    fake_size[1],
            }
        )

        per_method_valid[
            method
        ] += 1

        mask_areas.append(
            mask_area_frac
        )

    # ========================================================
    # 4. DUPLICATE CHECK
    # ========================================================

    sample_ids = [
        row["sample_id"]
        for row in forensic_gt_samples
    ]

    duplicates = (
        len(sample_ids)
        - len(set(sample_ids))
    )

    if duplicates:
        raise RuntimeError(
            f"Duplicate Forensic-GT "
            f"samples found: "
            f"{duplicates}"
        )

    # ========================================================
    # 5. SAVE FORENSIC_GT_MANIFEST
    # ========================================================

    manifest_fields = [
        "sample_id",
        "split",
        "method",
        "pair_id",
        "target_id",
        "source_id",
        "frame_id",
        "video_group",
        "fake_path",
        "pristine_path",
        "mask_path",
        "mask_area_px",
        "mask_area_frac",
        "width",
        "height",
    ]

    with FORENSIC_GT_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=
                manifest_fields,
        )

        writer.writeheader()

        writer.writerows(
            forensic_gt_samples
        )

    # Save exclusions.
    if excluded_rows:

        all_keys = set()

        for row in excluded_rows:
            all_keys.update(
                row.keys()
            )

        exclusion_fields = sorted(
            all_keys
        )

        with FORENSIC_GT_EXCLUDED.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=
                    exclusion_fields,
                extrasaction="ignore",
            )

            writer.writeheader()

            writer.writerows(
                excluded_rows
            )

    else:

        FORENSIC_GT_EXCLUDED.write_text(
            "",
            encoding="utf-8",
        )

    # ========================================================
    # 6. QA OVERLAYS
    # ========================================================

    section(
        "4. GENERATING GT-MASK QA OVERLAYS"
    )

    qa_counts = defaultdict(
        int
    )

    for row in forensic_gt_samples:

        method = row["method"]

        if (
            qa_counts[method]
            >= QA_PER_METHOD
        ):
            continue

        output = (
            FORENSIC_GT_QA_ROOT
            / (
                row["sample_id"]
                .replace("/", "__")
                + ".png"
            )
        )

        save_overlay(
            Path(
                row["fake_path"]
            ),
            Path(
                row["mask_path"]
            ),
            output,
        )

        qa_counts[
            method
        ] += 1

        print(
            output
        )

    # ========================================================
    # 7. FORENSIC_GT_SUMMARY
    # ========================================================

    section(
        "5. FINAL FORENSIC-GT DATA AUDIT"
    )

    summary = {
        "required_mask_videos":
            total_pairs,

        "valid_manifest_samples":
            len(
                forensic_gt_samples
            ),

        "excluded_samples":
            len(
                excluded_rows
            ),

        "duplicate_samples":
            duplicates,

        "valid_by_method":
            dict(
                per_method_valid
            ),

        "excluded_by_reason":
            dict(
                reason_counts
            ),

        "mask_area": {
            "min":
                (
                    float(
                        np.min(
                            mask_areas
                        )
                    )
                    if mask_areas
                    else None
                ),

            "median":
                (
                    float(
                        np.median(
                            mask_areas
                        )
                    )
                    if mask_areas
                    else None
                ),

            "mean":
                (
                    float(
                        np.mean(
                            mask_areas
                        )
                    )
                    if mask_areas
                    else None
                ),

            "max":
                (
                    float(
                        np.max(
                            mask_areas
                        )
                    )
                    if mask_areas
                    else None
                ),
        },

        "manifest":
            str(
                FORENSIC_GT_MANIFEST
            ),

        "excluded_csv":
            str(
                FORENSIC_GT_EXCLUDED
            ),

        "qa_root":
            str(
                FORENSIC_GT_QA_ROOT
            ),
    }

    FORENSIC_GT_SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()

    if (
        total_pairs == 280
        and duplicates == 0
        and len(
            forensic_gt_samples
        ) > 0
    ):

        print(
            "FORENSIC-GT DATA PREPARATION: PASS"
        )

        print(
            "Aligned official masks and "
            "exact fake/pristine pairs are "
            "ready for the controlled audit."
        )

    else:

        print(
            "FORENSIC-GT DATA PREPARATION: FAIL"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()