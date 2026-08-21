from __future__ import annotations

# ============================================================
# Environment setup MUST happen before importing libraries that
# may import transformers / initialize CUDA.
# ============================================================

import os

# Hugging Face deprecated TRANSFORMERS_CACHE.
# Preserve the existing cache location if one was configured.
_legacy_transformers_cache = os.environ.pop(
    "TRANSFORMERS_CACHE",
    None,
)

if _legacy_transformers_cache:
    os.environ.setdefault(
        "HF_HUB_CACHE",
        _legacy_transformers_cache,
    )

os.environ.setdefault(
    "HF_HOME",
    os.path.expanduser("~/.cache/huggingface"),
)

# Prevent obsolete W&B anonymous configuration from producing
# a warning.
os.environ.pop("WANDB_ANONYMOUS", None)


import argparse
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)

from rift.config import load_config
from rift.data.datamodule import RIFTDataModule
from rift.lightning.detector_module import DetectorLightningModule
from rift.logging_utils import build_logger


# ============================================================
# Runtime / performance settings
# ============================================================

# A100 Tensor Cores.
torch.set_float32_matmul_precision("high")


# ============================================================
# Suppress known harmless third-party warnings only.
#
# Do NOT globally disable warnings.
# ============================================================

warnings.filterwarnings(
    "ignore",
    message=r"Precision bf16-mixed is not supported by the model summary.*",
)

warnings.filterwarnings(
    "ignore",
    message=r"Grad strides do not match bucket view strides.*",
)


# ============================================================
# Helpers
# ============================================================

def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))

    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    return f"{minutes:02d}:{seconds:02d}"


def safe_name(value: str) -> str:
    value = str(value).strip()

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return value.strip("_") or "run"


def make_run_dir(cfg) -> str:
    """
    Create one unique directory per experiment.

    Important for DDP:
    rank 0 creates RIFT_RUN_ID before Lightning launches child
    processes. Child ranks inherit the same environment variable,
    so all four processes use the same directory.
    """

    wandb_cfg = getattr(
        cfg,
        "wandb",
        None,
    )

    configured_name = None

    if wandb_cfg is not None:
        configured_name = getattr(
            wandb_cfg,
            "name",
            None,
        )

    run_name = (
        str(configured_name)
        if configured_name
        else str(cfg.experiment.name)
    )

    run_name = safe_name(run_name)

    if "RIFT_RUN_ID" not in os.environ:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        os.environ["RIFT_RUN_ID"] = (
            f"{run_name}_{timestamp}"
        )

    run_id = os.environ["RIFT_RUN_ID"]

    run_dir = (
        Path(str(cfg.experiment.root_dir))
        / str(cfg.experiment.name)
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return str(run_dir)


# ============================================================
# Progress bar
# ============================================================

class RIFTProgressBar(TQDMProgressBar):
    """
    Display epochs as:

        Epoch 1/5
        Epoch 2/5

    instead of Lightning's zero-based Epoch 0, Epoch 1...
    """

    def on_train_epoch_start(
        self,
        trainer,
        pl_module,
    ):
        super().on_train_epoch_start(
            trainer,
            pl_module,
        )

        if self.train_progress_bar is not None:
            self.train_progress_bar.set_description(
                f"Epoch "
                f"{trainer.current_epoch + 1}/"
                f"{trainer.max_epochs}"
            )


# ============================================================
# Timing callback
# ============================================================

class TrainingTimer(pl.Callback):
    """
    Rank-zero-only training timer.

    Prevents four duplicate messages under DDP and correctly
    reports completed epochs.
    """

    def __init__(self):
        super().__init__()

        self.fit_start: float | None = None
        self.epoch_start: float | None = None
        self.completed_epochs = 0

    def on_fit_start(
        self,
        trainer,
        pl_module,
    ):
        self.fit_start = time.perf_counter()

    def on_train_epoch_start(
        self,
        trainer,
        pl_module,
    ):
        self.epoch_start = time.perf_counter()

    def on_validation_epoch_end(
        self,
        trainer,
        pl_module,
    ):
        # Ignore Lightning sanity validation.
        if trainer.sanity_checking:
            return

        # Only rank 0 should print.
        if not trainer.is_global_zero:
            return

        now = time.perf_counter()

        completed = int(
            trainer.current_epoch + 1
        )

        completed = min(
            completed,
            int(trainer.max_epochs),
        )

        self.completed_epochs = max(
            self.completed_epochs,
            completed,
        )

        epoch_time = 0.0

        if self.epoch_start is not None:
            epoch_time = (
                now - self.epoch_start
            )

        total_elapsed = 0.0

        if self.fit_start is not None:
            total_elapsed = (
                now - self.fit_start
            )

        avg_epoch_time = (
            total_elapsed
            / max(
                self.completed_epochs,
                1,
            )
        )

        remaining_epochs = max(
            int(trainer.max_epochs)
            - self.completed_epochs,
            0,
        )

        estimated_remaining = (
            avg_epoch_time
            * remaining_epochs
        )

        print(
            "\n"
            f"[Epoch "
            f"{self.completed_epochs}/"
            f"{trainer.max_epochs} complete] "
            f"epoch_time="
            f"{format_time(epoch_time)} | "
            f"total_elapsed="
            f"{format_time(total_elapsed)} | "
            f"estimated_remaining="
            f"{format_time(estimated_remaining)}"
        )

    def on_fit_end(
        self,
        trainer,
        pl_module,
    ):
        if not trainer.is_global_zero:
            return

        if self.fit_start is None:
            return

        total_time = (
            time.perf_counter()
            - self.fit_start
        )

        completed = min(
            self.completed_epochs,
            int(trainer.max_epochs),
        )

        print(
            "\n"
            "============================================\n"
            f"Training complete: "
            f"{completed}/"
            f"{trainer.max_epochs} epochs\n"
            f"Total wall-clock time: "
            f"{format_time(total_time)}\n"
            "============================================"
        )


# ============================================================
# Trainer
# ============================================================

def make_trainer(
    cfg,
    run_dir: str,
):
    callbacks = [
        ModelCheckpoint(
            dirpath=str(
                Path(run_dir)
                / "checkpoints"
            ),
            monitor=str(
                cfg.checkpoint.monitor
            ),
            mode=str(
                cfg.checkpoint.mode
            ),
            save_top_k=int(
                cfg.checkpoint.save_top_k
            ),
            save_last=bool(
                cfg.checkpoint.save_last
            ),
            filename=str(
                cfg.checkpoint.filename
            ),
            auto_insert_metric_name=False,
        ),

        LearningRateMonitor(
            logging_interval="epoch"
        ),

        RIFTProgressBar(
            refresh_rate=10
        ),

        TrainingTimer(),
    ]

    if bool(
        cfg.early_stopping.enable
    ):
        callbacks.append(
            EarlyStopping(
                monitor=str(
                    cfg.early_stopping.monitor
                ),
                mode=str(
                    cfg.early_stopping.mode
                ),
                patience=int(
                    cfg.early_stopping.patience
                ),
                min_delta=float(
                    cfg.early_stopping.min_delta
                ),
            )
        )

    logger = build_logger(
        cfg,
        run_dir,
    )

    t = cfg.trainer

    return pl.Trainer(
        default_root_dir=run_dir,

        accelerator=t.accelerator,
        devices=t.devices,
        strategy=t.strategy,

        precision=t.precision,

        max_epochs=int(
            t.max_epochs
        ),

        accumulate_grad_batches=int(
            t.accumulate_grad_batches
        ),

        gradient_clip_val=float(
            t.gradient_clip_val
        ),

        log_every_n_steps=int(
            t.log_every_n_steps
        ),

        deterministic=bool(
            t.deterministic
        ),

        benchmark=bool(
            t.benchmark
        ),

        num_sanity_val_steps=int(
            t.num_sanity_val_steps
        ),

        callbacks=callbacks,
        logger=logger,
    )


# ============================================================
# Entry point
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "train_detector_mixed.yaml"
        ),
    )

    parser.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Lightning checkpoint "
            "to resume from"
        ),
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "OmegaConf dotlist overrides, "
            "e.g. trainer.devices=4"
        ),
    )

    args = parser.parse_args(argv)

    cfg = load_config(
        args.config,
        args.overrides,
    )

    pl.seed_everything(
        int(cfg.seed),
        workers=True,
    )

    run_dir = make_run_dir(cfg)

    if (
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
        == "0"
    ):
        print(
            f"[RIFT] Run directory: "
            f"{run_dir}"
        )

    dm = RIFTDataModule(cfg)

    module = DetectorLightningModule(
        cfg
    )

    if not module.trainable_detector:
        raise SystemExit(
            "model.name=cift_external "
            "is frozen. "
            "Use validation/audit mode "
            "instead of detector training."
        )

    trainer = make_trainer(
        cfg,
        run_dir,
    )

    trainer.fit(
        module,
        datamodule=dm,
        ckpt_path=args.ckpt,
    )


if __name__ == "__main__":
    main()