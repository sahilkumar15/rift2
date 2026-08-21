# Migration notes: old RIFT -> locked RIFT v2

The old `rift-main (2).zip` was useful as an engineering reference for:

- external CIFT checkpoint loading;
- FF++ donor path conventions;
- configuration/checkpoint/logging organization;
- DDP-safe data handling ideas.

It should **not** be copied conceptually into the new paper without changes.

## Removed from the core scientific contract

The old implementation trained a PPO region-selection policy and used CIFT
identity-gap/internal features as part of the explanation objective/state. The locked
RIFT plan instead defines the central contribution as a **black-box score-access
forensic-specificity audit** of frozen detectors. Therefore:

- detector internals are not required by the FSS evaluator;
- donor-grounded CIFT identity gap is not a RIFT input;
- RL is not a main contribution;
- PPO/greedy/beam belong only to optional evidence discovery after the metric works;
- the detector and the auditor have separate training/evaluation lifecycles.

## Kept and cleaned

- CIFT external loader/checkpoint bridge;
- FF++ relation-aware filesystem logic;
- explicit train/validation separation;
- W&B logging;
- modular config-driven execution;
- AUC/EER and reproducible run structure.

## Main protocol warning

The supplied mixed-domain training YAML is preserved for detector experiments.
However, if Celeb-DF/DFD/WildDeepfake/DiffSwap are included in training, they cannot
simultaneously be described as zero-shot OOD targets for that trained detector.
For the paper's clean cross-dataset protocol, train with:

```bash
python -m rift.train --config configs/train_detector_mixed.yaml dataset.name=ffpp_rela
```

and reserve external datasets for OOD evaluation.
