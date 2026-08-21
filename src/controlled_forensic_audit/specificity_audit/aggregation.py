from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from forensic_audit.fss import fss_from_aggregates


REGION_ORDER = [
    "planted_shortcut",
    "gt_manipulation",
    "matched_background",
    "random_region",
]

BASE_METRICS = [
    "necessity",
    "sufficiency",
    "manipulation_reliance",
    "nuisance_instability",
]

REPORT_METRICS = [
    "necessity",
    "sufficiency",
    "faithfulness",
    "manipulation_reliance",
    "nuisance_instability",
    "fss",
]


def _harmonic(a: float, b: float, epsilon: float) -> float:
    return float(2.0 * a * b / (a + b + float(epsilon)))


def _aggregate_members(members: list[dict], epsilon: float) -> dict[str, float]:
    values = {
        key: float(np.mean([float(row[key]) for row in members]))
        for key in BASE_METRICS
    }
    values["faithfulness"] = _harmonic(
        values["necessity"],
        values["sufficiency"],
        epsilon,
    )
    values["fss"] = fss_from_aggregates(
        values["manipulation_reliance"],
        values["nuisance_instability"],
        epsilon=epsilon,
    )
    return values


def aggregate_results(cfg, per_sample_csv: Path, output_root: Path) -> dict[str, Path]:
    with per_sample_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Per-sample controlled audit CSV is empty")

    epsilon = float(cfg.audit.epsilon)

    # 1) Frame -> video group.
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["video_group"], row["evidence_region"])].append(row)

    group_rows: list[dict] = []
    for (method, video_group, region), members in grouped.items():
        group_rows.append(
            {
                "method": method,
                "video_group": video_group,
                "evidence_region": region,
                "forensic_gt": members[0]["forensic_gt"],
                "n_frames": len(members),
                **_aggregate_members(members, epsilon),
            }
        )

    grouped_csv = output_root / "forensic_specificity_video_group.csv"
    with grouped_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    # 2) Video groups -> method.
    method_buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in group_rows:
        method_buckets[(row["method"], row["evidence_region"])].append(row)

    method_rows: list[dict] = []
    for (method, region), members in sorted(method_buckets.items()):
        method_rows.append(
            {
                "method": method,
                "evidence_region": region,
                "forensic_gt": members[0]["forensic_gt"],
                "n_video_groups": len(members),
                **_aggregate_members(members, epsilon),
            }
        )

    method_csv = output_root / "forensic_specificity_by_method.csv"
    with method_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0].keys()))
        writer.writeheader()
        writer.writerows(method_rows)

    # 3) Equal-weight macro-average across manipulation methods.
    summary_rows: list[dict] = []
    for region in REGION_ORDER:
        members = [row for row in method_rows if row["evidence_region"] == region]
        if not members:
            continue
        summary_rows.append(
            {
                "evidence_region": region,
                "forensic_gt": members[0]["forensic_gt"],
                "n_methods": len(members),
                **_aggregate_members(members, epsilon),
            }
        )

    # 4) Stratified video-group bootstrap within each manipulation method.
    rng = np.random.default_rng(int(cfg.experiment.seed))
    n_boot = int(cfg.aggregation.bootstrap_replicates)
    alpha = float(cfg.aggregation.alpha)

    methods = sorted({row["method"] for row in group_rows})
    by_method_region: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in group_rows:
        by_method_region[(row["method"], row["evidence_region"])].append(row)

    ci_lookup: dict[str, dict[str, tuple[float, float]]] = {}

    for region in REGION_ORDER:
        bootstrap = {metric: [] for metric in REPORT_METRICS}

        for _ in range(n_boot):
            synthetic_method_rows = []
            for method in methods:
                population = by_method_region[(method, region)]
                if not population:
                    continue
                indices = rng.integers(0, len(population), size=len(population))
                sampled = [population[int(i)] for i in indices]
                synthetic_method_rows.append(_aggregate_members(sampled, epsilon))

            if not synthetic_method_rows:
                continue

            macro = _aggregate_members(synthetic_method_rows, epsilon)
            for metric in REPORT_METRICS:
                bootstrap[metric].append(macro[metric])

        region_ci = {}
        for metric in REPORT_METRICS:
            values = np.asarray(bootstrap[metric], dtype=np.float64)
            region_ci[metric] = (
                float(np.quantile(values, alpha / 2.0)),
                float(np.quantile(values, 1.0 - alpha / 2.0)),
            )
        ci_lookup[region] = region_ci

    for row in summary_rows:
        for metric in REPORT_METRICS:
            lo, hi = ci_lookup[row["evidence_region"]][metric]
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi

    summary_csv = output_root / "forensic_specificity_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    bootstrap_json = output_root / "forensic_specificity_bootstrap.json"
    bootstrap_payload = {
        "seed": int(cfg.experiment.seed),
        "bootstrap_replicates": n_boot,
        "alpha": alpha,
        "resampling_unit": "video_group",
        "stratified_by": "manipulation_method",
        "regions": {
            region: {
                metric: {"lower": bounds[0], "upper": bounds[1]}
                for metric, bounds in metrics.items()
            }
            for region, metrics in ci_lookup.items()
        },
    }
    bootstrap_json.write_text(
        json.dumps(bootstrap_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "video_group": grouped_csv,
        "by_method": method_csv,
        "summary": summary_csv,
        "bootstrap": bootstrap_json,
    }
