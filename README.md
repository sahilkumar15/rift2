# RIFT v2 — clean research codebase

This repository is organized around the **locked RIFT scientific story**:
RIFT audits an **already-trained frozen detector** using score access only. The core
quantity is Forensic Specificity Score (FSS), not reinforcement learning.

## 1. Scientific separation

There are two deliberately separate lifecycles:

1. **Detector lifecycle** — train/validate detector variants. This is where the
   mixed-domain YAML, PyTorch Lightning Trainer, checkpoints, AUC/EER, and W&B live.
2. **RIFT lifecycle** — freeze one detector checkpoint and audit it using paired
   authenticity-changing vs authenticity-preserving interventions. RIFT itself does
   not fine-tune the detector and does not need gradients or internal features.

This separation prevents the implementation from accidentally turning RIFT into a
new detector-training method.

## 2. Current project tree

```text
rift-v2/
├── configs/
│   ├── train_detector_mixed.yaml  # exact dataset structure/paths from the project
│   ├── validate_cift.yaml         # frozen CIFT score-access validation
│   └── rift_fss.yaml              # locked M/Q/FSS audit settings
├── scripts/
│   ├── preflight.sh
│   ├── train_detector.sh
│   └── validate_cift.sh
├── src/rift/
│   ├── config.py
│   ├── preflight.py
│   ├── train.py
│   ├── validate.py
│   ├── data/
│   │   ├── ffpp_relation.py       # real FF++ donor/target pairing
│   │   ├── csv_dataset.py         # CelebDF/DFD/Wild/DiffSwap CSV contract
│   │   ├── mixed.py               # weighted, auto-normalized domain mixture
│   │   ├── factory.py
│   │   ├── datamodule.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── convnext_binary.py     # clean trainable detector variant
│   │   ├── cift_external.py       # frozen bridge to existing CIFT repo/checkpoint
│   │   ├── tiny.py                # tests only
│   │   └── factory.py
│   ├── lightning/
│   │   └── detector_module.py
│   ├── metrics/
│   │   └── binary.py
│   └── audit/
│       ├── score.py               # calibrated-logit score wrapper
│       ├── interventions.py       # shared regional neutralization
│       ├── nuisances.py           # JPEG/blur/resize/gamma
│       └── fss.py                 # exact M, Q, FSS equations
└── tests/
```

## 3. Dataset behavior

`configs/train_detector_mixed.yaml` keeps the paths and structure supplied for:
FF++, Celeb-DF-v2, DFD, WildDeepfake, and DiffSwap.

The active raw weights currently sum to `1.231`, so they are normalized to:

- FF++ relation: **0.804224**
- Celeb-DF: **0.056864**
- DFD: **0.097482**
- WildDeepfake: **0.040617**
- DiffSwap: **0.000812**

`total_per_epoch: 100000` means exactly 100,000 training samples are drawn per
logical epoch according to those normalized proportions.

### FF++ relation loader

The loader supports the standard FF++ layout and standard split JSON files:

```text
original_sequences/youtube/c23/{images|frames}/<video_id>/...
manipulated_sequences/<method>/c23/{images|frames}/<source>_<target>/...
splits/train.json
splits/val.json
splits/test.json
```

For manipulated frames, the source identity is used to retrieve the donor frame.
For genuine frames, donor = target by construction.

### Non-FF++ CSV loaders

The generic loader accepts common aliases for image path and label columns. If a
CSV has a donor column it is used; otherwise `relation_valid=False` is explicit.

The YAML keys `use_sbi`, `sbi_prob`, and `freq_aug_prob` are preserved because they
belong to the existing CIFT data configuration. The clean generic loader **does not
silently implement an approximate SBI**. If exact SBI/Frequency-Blender synthesis
is required, wire the original implementation as a dataset plugin instead.

## 4. Metrics during detector training/validation

The first stage logs:

| Metric | Purpose |
|---|---|
| `train/loss`, `val/loss` | optimization and overfitting diagnosis |
| `val/auc` | primary threshold-free checkpoint metric |
| `val/eer` | error trade-off at the equal-error operating point |
| `val/ap` | useful when the real/fake distribution is imbalanced |
| `val/acc` | simple sanity metric, not the headline metric |
| `val/ece` | calibration error; relevant before later logit/FSS auditing |
| `val/brier` | probability calibration/sharpness sanity check |

Checkpoint selection uses **FF++ validation AUC only**. OOD datasets should not be
used for checkpoint selection; they are held for the later generalization study.

## 5. Install

```bash
cd /scratch/sahil/projects/img_deepfake/code/rift-v2
conda activate dif
pip install -e ".[logging,test]"
wandb login
```

## 6. Preflight

```bash
bash scripts/preflight.sh
```

To inspect only the normalized mix without checking Katz paths:

```bash
python -m rift.preflight --config configs/train_detector_mixed.yaml --skip-paths
```

## 7. Train a detector variant with Lightning

Single GPU:

```bash
python -m rift.train \
  --config configs/train_detector_mixed.yaml \
  trainer.devices=1 loader.batch_size=32
```

Four GPUs with Distributed Data Parallel (DDP):

```bash
python -m rift.train \
  --config configs/train_detector_mixed.yaml \
  trainer.devices=4 trainer.strategy=ddp loader.batch_size=32
```

Resume:

```bash
python -m rift.train \
  --config configs/train_detector_mixed.yaml \
  --ckpt experiments/rift_v2_detector_mixed/checkpoints/last.ckpt \
  trainer.devices=4 trainer.strategy=ddp
```

## 8. Validate the existing CIFT detector

The CIFT bridge loads the current external CIFT repo and checkpoint but keeps all
parameters frozen:

```bash
python -m rift.validate --config configs/validate_cift.yaml trainer.devices=1
```

This uses the **source-free detector score**. Donors are not fed into the deployed
score path, which is the right contract for the new score-access RIFT audit.

## 9. Locked RIFT metric implementation

`src/rift/audit/fss.py` implements:

- detector-level robust score scale `s_f = P95(logit) - P5(logit)`;
- Manipulation Reliance `M` from matched pristine/manipulated pairs;
- Nuisance Instability `Q` from authenticity-preserving JPEG/blur/resize/gamma;
- `FSS = 2 M (1-Q) / (M + (1-Q) + eps)`.

The detector interface is only:

```python
logits = score_fn(images)
```

No gradient, feature hook, attention map, donor input, or architecture-specific
internal state is required by FSS.

## 10. Important research rule for the next stage

Use pairable data (primarily FF++) for `M/Q/FSS`. Use Celeb-DF, DFD, WildDeepfake,
DiffSwap, DFDC/DF40, etc. for held-out OOD detector performance where pairing is
not valid. Do **not** invent matched pristine counterparts for an unpairable dataset.

RL should be added only as an optional region-search module after greedy/beam
baselines exist and only retained if it materially improves the search ablation.
