from __future__ import annotations

import warnings

from .csv_dataset import FaceCSVDataset
from .ffpp_relation import FFPPRelationDataset


def build_single_dataset(name: str, cfg, transform, *, split: str):
    name = str(name)
    if name == "ffpp_rela":
        return FFPPRelationDataset(cfg, transform, split=split, domain=name)

    split_csv = getattr(cfg, "split_csv", None)
    if not split_csv:
        raise ValueError(f"Dataset {name!r} requires split_csv in this clean loader")

    # These knobs are intentionally preserved for compatibility with the user's
    # existing CIFT data configs. The generic RIFT-v2 loader does not silently
    # implement an approximate SBI/Frequency-Blender algorithm; if exact SBI is
    # desired, it should be supplied as an explicit dataset plugin.
    if bool(getattr(cfg, "use_sbi", False)):
        warnings.warn(
            f"{name}: use_sbi/sbi_prob/freq_aug_prob are preserved in YAML but are not "
            "applied by the generic CSV loader. Use the original CIFT/SBI dataset plugin "
            "for exact paper-compatible synthesis rather than an approximation.",
            RuntimeWarning,
        )

    return FaceCSVDataset(
        str(split_csv),
        transform,
        domain=name,
        data_root=str(getattr(cfg, "data_root", "")) or None,
        strict_csv=bool(getattr(cfg, "strict_csv", False)),
        skip_bad_images=bool(getattr(cfg, "skip_bad_images", True)),
    )
