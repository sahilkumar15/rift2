from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .aggregation import aggregate_results
from .audit import run_forensic_specificity_audit
from .calibration import calibrate_score_scale
from .model import load_frozen_detector, resolve_device
from .reporting import write_latex_table
from .validation import validate_controlled_detector


STAGES = {"validate", "calibrate", "audit", "aggregate", "report", "all"}


def _load_cfg(path: str, overrides: list[str]):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required prior-stage result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the Controlled Forensic Specificity Audit used for the paper's controlled evidence table."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="all", choices=sorted(STAGES))
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config, args.overrides)
    output_root = Path(cfg.paths.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_root / "resolved_audit_config.yaml")

    device = resolve_device(str(cfg.runtime.device))
    module = None

    def detector():
        nonlocal module
        if module is None:
            module = load_frozen_detector(cfg.paths.detector_checkpoint, device)
        return module

    if args.stage in {"validate", "all"}:
        result = validate_controlled_detector(cfg, detector(), device, output_root)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(
                "Controlled detector FAILED validation gates. Audit stopped before Forensic-GT test evaluation."
            )

    if args.stage in {"calibrate", "all"}:
        if args.stage == "all":
            validation = _read_json(output_root / "detector_validation.json")
            if not validation.get("passed", False):
                raise SystemExit("Detector validation did not pass; refusing calibration/audit.")
        calibration = calibrate_score_scale(cfg, detector(), device, output_root)
        print(json.dumps(calibration, indent=2))

    if args.stage in {"audit", "all"}:
        validation = _read_json(output_root / "detector_validation.json")
        if not validation.get("passed", False):
            raise SystemExit("Detector validation did not pass; refusing Forensic-GT audit.")
        calibration = _read_json(output_root / "detector_calibration.json")
        per_sample = run_forensic_specificity_audit(
            cfg,
            detector(),
            device,
            output_root,
            score_scale=float(calibration["score_scale"]),
        )
        print(f"Per-sample audit: {per_sample}")

    if args.stage in {"aggregate", "all"}:
        per_sample = output_root / "forensic_specificity_per_sample.csv"
        outputs = aggregate_results(cfg, per_sample, output_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")

    if args.stage in {"report", "all"}:
        summary = output_root / "forensic_specificity_summary.csv"
        latex = write_latex_table(summary, output_root)
        print(f"LaTeX table: {latex}")


if __name__ == "__main__":
    main()
