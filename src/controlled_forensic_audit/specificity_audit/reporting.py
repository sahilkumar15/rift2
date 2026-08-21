from __future__ import annotations

import csv
from pathlib import Path


DISPLAY_NAMES = {
    "planted_shortcut": "Planted shortcut",
    "gt_manipulation": "GT manipulation region",
    "matched_background": "Matched background",
    "random_region": "Random region",
}


def _v(row: dict, key: str) -> str:
    return f"{float(row[key]):.3f}"


def write_latex_table(summary_csv: Path, output_root: Path) -> Path:
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    order = [
        "planted_shortcut",
        "gt_manipulation",
        "matched_background",
        "random_region",
    ]
    lookup = {row["evidence_region"]: row for row in rows}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controlled Forensic Specificity Audit on FF++ Forensic-GT. Higher is better for Necessity (Nec), Sufficiency (Suff), Faithfulness (Faith), manipulation reliance ($M$), and Forensic Specificity Score (FSS); lower is better for nuisance instability ($Q$). Aggregation is frame $\rightarrow$ video group $\rightarrow$ manipulation method, followed by an equal-weight macro-average across the four FF++ manipulation methods.}",
        r"\label{tab:controlled-forensic-specificity}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Evidence Region & Nec $\uparrow$ & Suff $\uparrow$ & Faith $\uparrow$ & $M$ $\uparrow$ & $Q$ $\downarrow$ & FSS $\uparrow$ & Forensic GT \\",
        r"\midrule",
    ]

    for region in order:
        row = lookup[region]
        lines.append(
            f"{DISPLAY_NAMES[region]} & "
            f"{_v(row, 'necessity')} & "
            f"{_v(row, 'sufficiency')} & "
            f"{_v(row, 'faithfulness')} & "
            f"{_v(row, 'manipulation_reliance')} & "
            f"{_v(row, 'nuisance_instability')} & "
            f"{_v(row, 'fss')} & "
            f"{'Yes' if int(row['forensic_gt']) else 'No'} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )

    output = output_root / "controlled_forensic_specificity_audit.tex"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
