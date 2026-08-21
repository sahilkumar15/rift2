from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl

from project_core.config import load_config
from detector_data.datamodule import RIFTDataModule
from detector_training.detector_module import DetectorLightningModule
from project_core.logging import build_logger


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="Lightning checkpoint for native detector; omit for external CIFT")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    pl.seed_everything(int(cfg.seed), workers=True)
    run_dir = str(Path(cfg.experiment.root_dir) / cfg.experiment.name)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    dm = RIFTDataModule(cfg)
    if args.ckpt:
        module = DetectorLightningModule.load_from_checkpoint(args.ckpt, cfg=cfg)
    else:
        module = DetectorLightningModule(cfg)

    t = cfg.trainer
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        accelerator=t.accelerator,
        devices=t.devices,
        strategy=t.strategy,
        precision=t.precision,
        num_sanity_val_steps=int(getattr(t, "num_sanity_val_steps", 0)),
        logger=build_logger(cfg, run_dir),
    )
    trainer.validate(module, datamodule=dm)


if __name__ == "__main__":
    main()
