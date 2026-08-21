from __future__ import annotations

import argparse
import os
from pathlib import Path

from rift.config import active_domain_weights, load_config


def _check(path: str, kind: str, errors: list[str]):
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    ok = p.is_file() if kind == "file" else p.is_dir()
    print(f"[{'OK' if ok else 'MISSING'}] {kind:4s} {p}")
    if not ok:
        errors.append(str(p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_detector_mixed.yaml")
    ap.add_argument("--skip-paths", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    print("\nActive mixed-domain weights (normalized):")
    for name, w in active_domain_weights(cfg).items():
        print(f"  {name:20s} {w:.6f}  ({100*w:6.3f}%)")

    if args.skip_paths:
        return

    errors: list[str] = []
    if str(cfg.dataset.name) == "mixed":
        for d in cfg.dataset.mixed.domains:
            dc = cfg.dataset[str(d.config_key)]
            _check(str(dc.data_root), "dir", errors)
            if hasattr(dc, "split_csv") and dc.split_csv:
                _check(str(dc.split_csv), "file", errors)
    else:
        dc = cfg.dataset[str(cfg.dataset.name)]
        _check(str(dc.data_root), "dir", errors)

    vdc = cfg.val_dataset[str(cfg.val_dataset.name)]
    _check(str(vdc.data_root), "dir", errors)

    if errors:
        raise SystemExit(f"\nPreflight failed: {len(errors)} required paths are missing.")
    print("\nPreflight passed.")


if __name__ == "__main__":
    main()
