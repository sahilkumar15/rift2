# RIFT v2 source and migration notes

This clean tree was reconstructed from the supplied main RIFT repository and the supplied Controlled Forensic Specificity Audit patch.

## Structural migration

The former `src/rift/` namespace was removed intentionally. Importable code now lives directly under `src/` in purpose-specific packages:

- `project_core`
- `detector_data`
- `detector_models`
- `detector_metrics`
- `detector_training`
- `forensic_audit`
- `controlled_forensic_audit`

All former `rift.*` imports and `python -m rift.*` launch commands were rewritten accordingly.

The former internal `src/rift/lightning/` folder was **not** flattened to `src/lightning/`, because that would shadow the installed third-party `lightning` package. It is now `src/detector_training/`.

## Scientific separation

The old detector/RL engineering code remains useful as implementation context, but the locked RIFT scientific contract is a black-box score-access forensic-specificity audit of frozen detectors.

Therefore:

- detector internals are not required by FSS;
- donor-grounded CIFT identity gap is not a RIFT input;
- RL is not a required RIFT contribution;
- detector training and forensic auditing have separate lifecycles;
- the controlled shortcut experiment is a validation experiment, not the definition of RIFT itself.

## Controlled paper experiment naming

Source files do not use `table1` in their names. The semantic experiment name is **Controlled Forensic Specificity Audit**. Its report may appear as Table 1 in one paper revision and move later without requiring code renaming.

## Protocol warning

If external datasets are included in detector training, they cannot simultaneously be described as zero-shot OOD test sets for that detector. For a strict cross-dataset protocol, train only on FF++ and reserve external datasets for evaluation.
