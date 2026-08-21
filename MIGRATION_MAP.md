# Source-layout migration map

The clean repository removes the `src/rift/` namespace completely.

| Old path | New path |
|---|---|
| `src/rift/config.py` | `src/project_core/config.py` |
| `src/rift/logging_utils.py` | `src/project_core/logging.py` |
| `src/rift/utils/seed.py` | `src/project_core/seed.py` |
| `src/rift/data/*` | `src/detector_data/*` |
| `src/rift/models/*` | `src/detector_models/*` |
| `src/rift/metrics/*` | `src/detector_metrics/*` |
| `src/rift/lightning/detector_module.py` | `src/detector_training/detector_module.py` |
| `src/rift/train.py` | `src/detector_training/train.py` |
| `src/rift/validate.py` | `src/detector_training/validate.py` |
| `src/rift/preflight.py` | `src/detector_training/preflight.py` |
| `src/rift/audit/*` | `src/forensic_audit/*` |
| `src/rift/controlled_forensic_audit/*` | `src/controlled_forensic_audit/*` |
| `src/rift/controlled_forensic_audit/evaluation/*` | `src/controlled_forensic_audit/specificity_audit/*` |

The internal package is **not** renamed to top-level `lightning`, because that would shadow the installed PyTorch Lightning package.

## Import migration

Examples:

```python
# old
from rift.audit.fss import compute_fss
from rift.controlled_forensic_audit import ControlledForensicDetectorModule
from rift.data.datamodule import RIFTDataModule
from rift.lightning.detector_module import DetectorLightningModule

# new
from forensic_audit.fss import compute_fss
from controlled_forensic_audit import ControlledForensicDetectorModule
from detector_data.datamodule import RIFTDataModule
from detector_training.detector_module import DetectorLightningModule
```

## Module launch migration

```text
python -m rift.train
→ python -m detector_training.train

python -m rift.validate
→ python -m detector_training.validate

python -m rift.preflight
→ python -m detector_training.preflight

python -m rift.controlled_forensic_audit.train
→ python -m controlled_forensic_audit.train

python -m rift.controlled_forensic_audit.evaluation.cli
→ python -m controlled_forensic_audit.specificity_audit.cli
```

The recommended public interface is the shell launchers under `scripts/` rather than invoking modules directly.
