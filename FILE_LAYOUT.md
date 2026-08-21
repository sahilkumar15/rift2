# Complete relative file layout

This repository intentionally has **no `src/rift/` package**. Importable packages live directly under `src/`.

## Configuration

- `configs/controlled_forensic_audit/detector.yaml` — controlled shortcut detector training configuration.
- `configs/controlled_forensic_audit/audit.yaml` — frozen-detector Controlled Forensic Specificity Audit configuration.
- `configs/rift_fss.yaml` — generic RIFT FSS configuration retained from the original repository.
- `configs/train_detector_mixed.yaml` — generic mixed-domain detector training configuration.
- `configs/validate_cift.yaml` — external CIFT bridge validation configuration.

## Shell launchers

The `scripts/` directory contains shell launchers only.

- `scripts/preflight.sh` — generic training preflight.
- `scripts/train_detector.sh` — generic detector training.
- `scripts/validate_cift.sh` — CIFT bridge validation.
- `scripts/train_controlled_detector.sh` — controlled shortcut detector training.
- `scripts/prepare_forensic_gt_data.sh` — prepare FF++ Forensic-GT data/manifest.
- `scripts/run_controlled_forensic_specificity_audit.sh` — validate, calibrate, audit, aggregate, and report the controlled forensic-specificity experiment.

## Shared project infrastructure

- `src/project_core/__init__.py`
- `src/project_core/config.py` — OmegaConf loading/overrides and domain-weight helpers.
- `src/project_core/logging.py` — Lightning/W&B logger construction.
- `src/project_core/seed.py` — reproducibility helper.

## Generic detector data

- `src/detector_data/__init__.py`
- `src/detector_data/csv_dataset.py`
- `src/detector_data/datamodule.py`
- `src/detector_data/factory.py`
- `src/detector_data/ffpp_relation.py`
- `src/detector_data/mixed.py`
- `src/detector_data/transforms.py`

## Generic detector models

- `src/detector_models/__init__.py`
- `src/detector_models/cift_external.py`
- `src/detector_models/convnext_binary.py`
- `src/detector_models/factory.py`
- `src/detector_models/tiny.py`

## Generic detector metrics

- `src/detector_metrics/__init__.py`
- `src/detector_metrics/binary.py`

## Generic detector lifecycle

- `src/detector_training/__init__.py`
- `src/detector_training/detector_module.py` — Lightning detector module.
- `src/detector_training/train.py` — generic detector training CLI.
- `src/detector_training/validate.py` — generic detector validation CLI.
- `src/detector_training/preflight.py` — generic preflight CLI.

The old internal folder `src/rift/lightning/` is **not** renamed to top-level `src/lightning/`, because that would shadow the installed `lightning` package. `detector_training` avoids that collision.

## Generic RIFT forensic audit primitives

- `src/forensic_audit/__init__.py`
- `src/forensic_audit/score.py` — score-access adapter contract.
- `src/forensic_audit/interventions.py` — shared region neutralization.
- `src/forensic_audit/nuisances.py` — authenticity-preserving nuisance transforms.
- `src/forensic_audit/fss.py` — robust score scale, manipulation reliance M, nuisance instability Q, and FSS.

## Controlled detector / Forensic-GT preparation

- `src/controlled_forensic_audit/__init__.py`
- `src/controlled_forensic_audit/data.py` — full/balanced FF++ binary loader and controlled detector dataset.
- `src/controlled_forensic_audit/detector.py` — ConvNeXt-Tiny controlled detector.
- `src/controlled_forensic_audit/shortcut.py` — deterministic planted checkerboard shortcut.
- `src/controlled_forensic_audit/train.py` — controlled detector training, DDP, early stopping, resume, checkpoints, W&B.
- `src/controlled_forensic_audit/prepare_forensic_gt.py` — exact FF++ fake/pristine/mask manifest preparation.

## Controlled Forensic Specificity Audit

This folder is the semantic implementation behind the controlled paper result; it is intentionally not named `table1`.

- `src/controlled_forensic_audit/specificity_audit/__init__.py`
- `src/controlled_forensic_audit/specificity_audit/cli.py` — staged audit orchestrator.
- `src/controlled_forensic_audit/specificity_audit/model.py` — frozen checkpoint loading and score function.
- `src/controlled_forensic_audit/specificity_audit/data.py` — validation loader and exact Forensic-GT paired dataset.
- `src/controlled_forensic_audit/specificity_audit/validation.py` — R0/R1/F0/F1 controlled detector validation gates.
- `src/controlled_forensic_audit/specificity_audit/calibration.py` — validation-only P5/P95 score-scale calibration.
- `src/controlled_forensic_audit/specificity_audit/regions.py` — planted shortcut, GT manipulation, matched-background, and random controls.
- `src/controlled_forensic_audit/specificity_audit/metrics.py` — necessity, sufficiency, faithfulness, M, Q, and FSS per sample.
- `src/controlled_forensic_audit/specificity_audit/audit.py` — frozen-detector Forensic-GT audit loop.
- `src/controlled_forensic_audit/specificity_audit/aggregation.py` — frame → video-group → manipulation-method macro aggregation and stratified bootstrap.
- `src/controlled_forensic_audit/specificity_audit/reporting.py` — paper-ready LaTeX controlled evidence table.

## Tests

- `tests/test_config_and_mix.py`
- `tests/test_ffpp_loader.py`
- `tests/test_fss.py`
- `tests/test_lightning_smoke.py`
- `tests/test_controlled_specificity_audit.py`
- `tests/test_repository_layout.py`

## Generated runtime outputs

These are intentionally **not bundled as source code** and are ignored by Git.

Controlled detector checkpoint expected after training:

`experiments/controlled_forensic_audit_2/controlled_detector/checkpoints/controlled_detector.ckpt`

Audit outputs expected after the full controlled audit:

`results/controlled_forensic_specificity_audit/resolved_audit_config.yaml`

`results/controlled_forensic_specificity_audit/detector_validation.json`

`results/controlled_forensic_specificity_audit/detector_calibration.json`

`results/controlled_forensic_specificity_audit/forensic_specificity_per_sample.csv`

`results/controlled_forensic_specificity_audit/forensic_specificity_video_group.csv`

`results/controlled_forensic_specificity_audit/forensic_specificity_by_method.csv`

`results/controlled_forensic_specificity_audit/forensic_specificity_summary.csv`

`results/controlled_forensic_specificity_audit/forensic_specificity_bootstrap.json`

`results/controlled_forensic_specificity_audit/controlled_forensic_specificity_audit.tex`
