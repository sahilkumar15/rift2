# RIFT v2 — flattened, modular research codebase

This repository keeps the scientific separation required by the RIFT paper:

- detector training is one lifecycle;
- RIFT is a score-access audit of a frozen detector;
- the controlled shortcut experiment is a validation experiment for forensic specificity;
- paper table numbering is not encoded into source filenames.

The old `src/rift/...` namespace has been removed. All importable packages now live directly under `src/` with collision-safe, purpose-specific names.

## Repository layout

```text
rift2/
├── configs/
│   ├── controlled_forensic_audit/
│   │   ├── detector.yaml
│   │   └── audit.yaml
│   ├── rift_fss.yaml
│   ├── train_detector_mixed.yaml
│   └── validate_cift.yaml
│
├── scripts/                         # shell launchers only
│   ├── preflight.sh
│   ├── train_detector.sh
│   ├── validate_cift.sh
│   ├── train_controlled_detector.sh
│   ├── prepare_forensic_gt_data.sh
│   └── run_controlled_forensic_specificity_audit.sh
│
├── src/
│   ├── project_core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── seed.py
│   │
│   ├── detector_data/
│   │   ├── __init__.py
│   │   ├── csv_dataset.py
│   │   ├── datamodule.py
│   │   ├── factory.py
│   │   ├── ffpp_relation.py
│   │   ├── mixed.py
│   │   └── transforms.py
│   │
│   ├── detector_models/
│   │   ├── __init__.py
│   │   ├── cift_external.py
│   │   ├── convnext_binary.py
│   │   ├── factory.py
│   │   └── tiny.py
│   │
│   ├── detector_metrics/
│   │   ├── __init__.py
│   │   └── binary.py
│   │
│   ├── detector_training/
│   │   ├── __init__.py
│   │   ├── detector_module.py
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── preflight.py
│   │
│   ├── forensic_audit/              # generic RIFT score-access primitives
│   │   ├── __init__.py
│   │   ├── fss.py
│   │   ├── interventions.py
│   │   ├── nuisances.py
│   │   └── score.py
│   │
│   └── controlled_forensic_audit/   # controlled proof experiment
│       ├── __init__.py
│       ├── data.py
│       ├── detector.py
│       ├── shortcut.py
│       ├── train.py
│       ├── prepare_forensic_gt.py
│       └── specificity_audit/
│           ├── __init__.py
│           ├── cli.py
│           ├── model.py
│           ├── data.py
│           ├── validation.py
│           ├── calibration.py
│           ├── regions.py
│           ├── metrics.py
│           ├── audit.py
│           ├── aggregation.py
│           └── reporting.py
│
├── experiments/                     # checkpoints; not source code
├── results/                         # audit outputs; not source code
└── tests/
```

The internal package formerly named `lightning` was intentionally renamed to `detector_training`. A top-level package named `lightning` would shadow the third-party PyTorch Lightning package after flattening `src/rift/`.

## Install

```bash
cd /scratch/sahil/projects/img_deepfake/code/rift2
conda activate dif
python -m pip install -e ".[logging,test]"
```

## Main detector lifecycle

Preflight:

```bash
./scripts/preflight.sh
```

Train generic detector:

```bash
CUDA_VISIBLE_DEVICES=6,7,1,0 \
./scripts/train_detector.sh
```

Validate external CIFT bridge:

```bash
CUDA_VISIBLE_DEVICES=6 \
./scripts/validate_cift.sh
```

## Controlled detector lifecycle

Train the shortcut-reliance calibration detector:

```bash
CUDA_VISIBLE_DEVICES=6,7,1,0 \
./scripts/train_controlled_detector.sh
```

The canonical selected checkpoint is expected at:

```text
experiments/controlled_forensic_audit_2/
└── controlled_detector/
    └── checkpoints/
        └── controlled_detector.ckpt
```

`last.ckpt` is for resuming optimization. The controlled audit uses `controlled_detector.ckpt`.

## Controlled Forensic Specificity Audit

The proper experiment name is **Controlled Forensic Specificity Audit**. The source code does not use `table1` in filenames because table numbering can change during paper revision.

Configuration:

```text
configs/controlled_forensic_audit/audit.yaml
```

The complete pipeline is:

```text
frozen controlled detector
        ↓
2×2 validation on FF++ validation
        ↓
validation-only score-scale calibration
        ↓
Forensic-GT paired audit
        ↓
frame → video-group → manipulation-method aggregation
        ↓
stratified video-group bootstrap
        ↓
LaTeX controlled-evidence table
```

Run every stage:

```bash
CUDA_VISIBLE_DEVICES=6 \
./scripts/run_controlled_forensic_specificity_audit.sh all
```

Or run one stage at a time:

```bash
CUDA_VISIBLE_DEVICES=6 ./scripts/run_controlled_forensic_specificity_audit.sh validate
CUDA_VISIBLE_DEVICES=6 ./scripts/run_controlled_forensic_specificity_audit.sh calibrate
CUDA_VISIBLE_DEVICES=6 ./scripts/run_controlled_forensic_specificity_audit.sh audit
./scripts/run_controlled_forensic_specificity_audit.sh aggregate
./scripts/run_controlled_forensic_specificity_audit.sh report
```

Expected result files:

```text
results/controlled_forensic_specificity_audit/
├── resolved_audit_config.yaml
├── detector_validation.json
├── detector_calibration.json
├── forensic_specificity_per_sample.csv
├── forensic_specificity_video_group.csv
├── forensic_specificity_by_method.csv
├── forensic_specificity_summary.csv
├── forensic_specificity_bootstrap.json
└── controlled_forensic_specificity_audit.tex
```

## Controlled evidence regions

Every valid Forensic-GT pair is audited using four predefined evidence regions:

1. `planted_shortcut` — known detector shortcut, not forensic ground truth;
2. `gt_manipulation` — official aligned FF++ manipulation mask;
3. `matched_background` — area/shape-preserving translated control selected to match pristine-region statistics;
4. `random_region` — deterministic area/shape-preserving negative control.

The planted shortcut is inserted identically into both members of the matched fake/pristine pair. This is the key controlled condition: generic detector faithfulness can still react to the shortcut, while pair-based manipulation reliance tests whether it actually explains the authenticity-changing difference.

## RIFT score contract

The generic implementation is in `src/forensic_audit/` and requires detector score access only:

```python
logits = score_fn(images)
```

No gradients, feature hooks, attention maps, donor input, or detector-specific internal state are required by the FSS computation.

The locked quantities are:

- robust calibration scale `s_f = P95(g(D_cal)) - P5(g(D_cal))`;
- manipulation reliance `M`;
- nuisance instability `Q` under JPEG/blur/resize/gamma;
- `FSS = 2 M (1-Q) / (M + (1-Q) + eps)`.

Calibration uses validation data only. Forensic-GT is opened only after the frozen detector passes the validation gates.

## Statistical aggregation

The controlled result is not obtained by treating frames as independent observations. The implementation aggregates:

```text
frame
  ↓
video group = manipulation method / pair ID
  ↓
manipulation method
  ↓
equal-weight macro average over FF++ manipulation methods
```

Bootstrap resampling is performed at video-group level within manipulation method.

## Test

```bash
pytest -q
```

The clean flattened package includes unit tests for configuration/mixing, FF++ pairing, generic FSS, Lightning smoke training, flattened package layout, evidence-region construction, and score-only FSS ingredients.
