#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from controlled_forensic_audit import (
    ControlledForensicDataset,
    ControlledForensicDetectorModule,
    load_binary_rows,
)


# ----------------------------------------------------------------------
# Progress callbacks
# ----------------------------------------------------------------------


class EpochProgressCallback(L.Callback):
    """Print one-based human-readable epoch banners on global rank zero."""

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del pl_module

        if not trainer.is_global_zero:
            return

        current = trainer.current_epoch + 1
        total = trainer.max_epochs

        print(
            "\n"
            + "=" * 72
            + f"\nEPOCH {current}/{total}"
            + "\n"
            + "=" * 72,
            flush=True,
        )

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del pl_module

        if not trainer.is_global_zero:
            return

        current = trainer.current_epoch + 1
        total = trainer.max_epochs

        print(
            f"\nCompleted epoch {current}/{total}",
            flush=True,
        )


class EpochAwareTQDMProgressBar(TQDMProgressBar):
    """
    Display one-based epoch numbers in Lightning's live training bar.

    Instead of:
        Epoch 0: 23%

    display:
        EPOCH 1/100: 23%
    """

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:

        super().on_train_epoch_start(
            trainer,
            pl_module,
        )

        if self.train_progress_bar is not None:
            self.train_progress_bar.set_description(
                f"EPOCH {trainer.current_epoch + 1}/{trainer.max_epochs}"
            )


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------


def is_local_rank_zero() -> bool:
    """Rank-zero helper usable before Trainer initialization."""

    return int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    ) == 0


def get_config_value(
    config: DictConfig,
    key: str,
    default,
):
    """Read an optional OmegaConf value with a default."""

    return OmegaConf.select(
        config,
        key,
        default=default,
    )


def load_config(
    config_path: str,
    overrides: list[str],
) -> DictConfig:
    """Load YAML, apply overrides, and resolve interpolation."""

    config = OmegaConf.load(
        config_path
    )

    if overrides:
        config = OmegaConf.merge(
            config,
            OmegaConf.from_dotlist(
                overrides
            ),
        )

    OmegaConf.resolve(
        config
    )

    return config


def resolve_resume_checkpoint(
    config: DictConfig,
    checkpoint_root: Path,
) -> Optional[str]:
    """
    Resolve training-resume checkpoint.

    Modes:
        auto:
            Resume checkpoint_root/last.ckpt if available.

        path:
            Resume resume.checkpoint_path.

        none:
            Start from scratch.
    """

    enabled = bool(
        get_config_value(
            config,
            "resume.enabled",
            True,
        )
    )

    if not enabled:
        return None

    mode = str(
        get_config_value(
            config,
            "resume.mode",
            "auto",
        )
    ).strip().lower()

    if mode == "none":
        return None

    if mode == "auto":
        checkpoint = (
            checkpoint_root
            / "last.ckpt"
        )

        if checkpoint.is_file():
            return str(
                checkpoint.resolve()
            )

        return None

    if mode == "path":
        configured_path = (
            get_config_value(
                config,
                "resume.checkpoint_path",
                None,
            )
        )

        if not configured_path:
            raise ValueError(
                "resume.mode='path' requires "
                "resume.checkpoint_path."
            )

        checkpoint = Path(
            str(configured_path)
        ).expanduser().resolve()

        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Resume checkpoint does not exist: "
                f"{checkpoint}"
            )

        return str(
            checkpoint
        )

    raise ValueError(
        f"Unsupported resume.mode={mode!r}. "
        "Use 'auto', 'path', or 'none'."
    )


def resolve_accelerator(
    config: DictConfig,
) -> str:
    """Resolve accelerator using CUDA auto-detection."""

    configured = str(
        get_config_value(
            config,
            "trainer.accelerator",
            "auto",
        )
    ).strip().lower()

    if configured == "auto":
        return (
            "gpu"
            if torch.cuda.is_available()
            else "cpu"
        )

    return configured


def resolve_trainer_devices(
    config: DictConfig,
    accelerator: str,
):
    """Resolve number of Lightning devices."""

    configured = (
        get_config_value(
            config,
            "trainer.devices",
            "auto",
        )
    )

    if accelerator == "cpu":
        if (
            str(configured)
            .lower()
            == "auto"
        ):
            return 1

        return int(
            configured
        )

    visible_count = (
        torch.cuda.device_count()
    )

    if (
        str(configured)
        .lower()
        == "auto"
    ):
        if visible_count < 1:
            raise RuntimeError(
                "GPU accelerator selected but "
                "no CUDA devices are visible."
            )

        return visible_count

    devices = int(
        configured
    )

    if devices < 1:
        raise ValueError(
            "trainer.devices must be at least 1."
        )

    if devices > visible_count:
        raise RuntimeError(
            f"Requested {devices} GPUs but only "
            f"{visible_count} are visible."
        )

    return devices


def resolve_strategy(
    config: DictConfig,
    accelerator: str,
    devices,
) -> str:
    """Automatically select DDP when multiple GPUs are used."""

    configured = str(
        get_config_value(
            config,
            "trainer.strategy",
            "auto",
        )
    ).strip().lower()

    if configured != "auto":
        return configured

    if (
        accelerator == "gpu"
        and isinstance(
            devices,
            int,
        )
        and devices > 1
    ):
        return "ddp"

    return "auto"


def count_binary_rows(
    rows: list[dict],
) -> tuple[int, int]:
    """Return real and fake counts."""

    real = 0
    fake = 0

    for row in rows:

        label = int(
            float(
                row["label"]
            )
        )

        if label == 0:
            real += 1

        elif label == 1:
            fake += 1

        else:
            raise ValueError(
                f"Invalid binary label: {label}"
            )

    return real, fake


# ----------------------------------------------------------------------
# Data construction
# ----------------------------------------------------------------------


def build_rows(
    config: DictConfig,
    seed: int,
) -> tuple[
    list[dict],
    list[dict],
    str,
    str,
]:
    """Load configured train and validation rows."""

    train_sampling_mode = str(
        get_config_value(
            config,
            "data.train_sampling_mode",
            "balanced",
        )
    ).strip().lower()

    val_sampling_mode = str(
        get_config_value(
            config,
            "data.val_sampling_mode",
            "balanced",
        )
    ).strip().lower()

    train_rows = load_binary_rows(
        config.paths.train_csv,
        seed=seed,
        sampling_mode=(
            train_sampling_mode
        ),
        max_per_class=(
            get_config_value(
                config,
                "data.train_max_per_class",
                None,
            )
        ),
    )

    validation_rows = load_binary_rows(
        config.paths.val_csv,
        seed=seed + 1,
        sampling_mode=(
            val_sampling_mode
        ),
        max_per_class=(
            get_config_value(
                config,
                "data.val_max_per_class",
                None,
            )
        ),
    )

    return (
        train_rows,
        validation_rows,
        train_sampling_mode,
        val_sampling_mode,
    )


def build_dataloaders(
    config: DictConfig,
    train_rows: list[dict],
    validation_rows: list[dict],
    seed: int,
) -> tuple[
    DataLoader,
    DataLoader,
]:
    """Build train and validation datasets/DataLoaders."""

    image_size = int(
        config.data.image_size
    )

    train_dataset = (
        ControlledForensicDataset(
            train_rows,
            image_size=image_size,
            seed=seed,
            training=True,
            fake_shortcut_probability=float(
                config.shortcut
                .fake_presence_probability
            ),
            real_shortcut_probability=float(
                config.shortcut
                .real_presence_probability
            ),
        )
    )

    validation_dataset = (
        ControlledForensicDataset(
            validation_rows,
            image_size=image_size,
            seed=seed + 1,
            training=False,
            fake_shortcut_probability=float(
                config.shortcut
                .fake_presence_probability
            ),
            real_shortcut_probability=float(
                config.shortcut
                .real_presence_probability
            ),
        )
    )

    batch_size = int(
        config.loader.batch_size
    )

    num_workers = int(
        config.loader.num_workers
    )

    pin_memory = bool(
        config.loader.pin_memory
    )

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (
            num_workers > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=bool(
            get_config_value(
                config,
                "loader.drop_last_train",
                False,
            )
        ),
        **common_kwargs,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    return (
        train_loader,
        validation_loader,
    )


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------


def build_module(
    config: DictConfig,
) -> ControlledForensicDetectorModule:
    """Construct controlled forensic detector module."""

    return (
        ControlledForensicDetectorModule(
            pretrained=bool(
                config.model.pretrained
            ),
            learning_rate=float(
                config.optimization
                .learning_rate
            ),
            weight_decay=float(
                config.optimization
                .weight_decay
            ),
            clean_supervision_weight=float(
                config.optimization
                .clean_supervision_weight
            ),
            shortcut_size_px=int(
                config.shortcut.size_px
            ),
            shortcut_top_px=int(
                config.shortcut.top_px
            ),
            shortcut_left_px=int(
                config.shortcut.left_px
            ),
            shortcut_tile_px=int(
                config.shortcut.tile_px
            ),
            shortcut_low_value=float(
                config.shortcut.low_value
            ),
            shortcut_high_value=float(
                config.shortcut.high_value
            ),
        )
    )


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------


def build_callbacks(
    config: DictConfig,
    checkpoint_root: Path,
) -> tuple[
    list[L.Callback],
    ModelCheckpoint,
]:
    """Build checkpoint, early-stop, and progress callbacks."""

    monitor = str(
        get_config_value(
            config,
            "checkpoint.monitor",
            "val_control_score",
        )
    )

    mode = str(
        get_config_value(
            config,
            "checkpoint.mode",
            "max",
        )
    )

    filename = str(
        get_config_value(
            config,
            "checkpoint.filename",
            (
                "controlled-detector-"
                "{epoch:02d}-"
                "{val_control_score:.4f}"
            ),
        )
    )

    checkpoint_callback = (
        ModelCheckpoint(
            dirpath=checkpoint_root,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=int(
                get_config_value(
                    config,
                    "checkpoint.save_top_k",
                    2,
                )
            ),
            save_last=bool(
                get_config_value(
                    config,
                    "checkpoint.save_last",
                    True,
                )
            ),
            every_n_epochs=int(
                get_config_value(
                    config,
                    "checkpoint.every_n_epochs",
                    1,
                )
            ),

            # val_control_score is generated
            # during validation.
            save_on_train_epoch_end=False,
        )
    )

    callbacks: list[L.Callback] = [
        checkpoint_callback,
    ]

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    if bool(
        get_config_value(
            config,
            "early_stopping.enabled",
            True,
        )
    ):

        callbacks.append(
            EarlyStopping(
                monitor=str(
                    get_config_value(
                        config,
                        "early_stopping.monitor",
                        monitor,
                    )
                ),
                mode=str(
                    get_config_value(
                        config,
                        "early_stopping.mode",
                        mode,
                    )
                ),
                patience=int(
                    get_config_value(
                        config,
                        "early_stopping.patience",
                        15,
                    )
                ),
                min_delta=float(
                    get_config_value(
                        config,
                        "early_stopping.min_delta",
                        0.001,
                    )
                ),
                verbose=bool(
                    get_config_value(
                        config,
                        "early_stopping.verbose",
                        True,
                    )
                ),
                strict=bool(
                    get_config_value(
                        config,
                        "early_stopping.strict",
                        True,
                    )
                ),
                check_finite=bool(
                    get_config_value(
                        config,
                        "early_stopping.check_finite",
                        True,
                    )
                ),

                # Important:
                # val_control_score only exists
                # after validation.
                check_on_train_epoch_end=False,
            )
        )

    # ------------------------------------------------------------------
    # Human-readable epoch banner
    # ------------------------------------------------------------------

    if bool(
        get_config_value(
            config,
            "progress.show_epoch_counter",
            True,
        )
    ):

        callbacks.append(
            EpochProgressCallback()
        )

    # ------------------------------------------------------------------
    # Custom live progress bar
    # ------------------------------------------------------------------

    if bool(
        get_config_value(
            config,
            "progress.enable_progress_bar",
            True,
        )
    ):

        callbacks.append(
            EpochAwareTQDMProgressBar(
                refresh_rate=int(
                    get_config_value(
                        config,
                        "progress.refresh_rate",
                        1,
                    )
                )
            )
        )

    return (
        callbacks,
        checkpoint_callback,
    )


# ----------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------


def build_logger(
    config: DictConfig,
    output_root: Path,
):
    """Build W&B logger with CSV fallback."""

    csv_logger = CSVLogger(
        save_dir=str(
            output_root
            / "logs"
        ),
        name="lightning",
    )

    if not bool(
        get_config_value(
            config,
            "wandb.enabled",
            False,
        )
    ):
        return csv_logger

    try:
        from lightning.pytorch.loggers import (
            WandbLogger,
        )

        return WandbLogger(
            project=str(
                config.wandb.project
            ),
            name=str(
                config.wandb.name
            ),
            save_dir=str(
                output_root
            ),
        )

    except Exception as error:

        if is_local_rank_zero():
            print(
                "[WARN] W&B unavailable; "
                "using CSV logger instead: "
                f"{error}",
                flush=True,
            )

        return csv_logger


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------


def build_trainer(
    config: DictConfig,
    callbacks: list[L.Callback],
    logger,
) -> L.Trainer:
    """Construct Lightning Trainer from config."""

    accelerator = (
        resolve_accelerator(
            config
        )
    )

    devices = (
        resolve_trainer_devices(
            config,
            accelerator,
        )
    )

    strategy = (
        resolve_strategy(
            config,
            accelerator,
            devices,
        )
    )

    if is_local_rank_zero():

        print(
            "\n"
            + "=" * 72
            + "\nTRAINER"
            + "\n"
            + "=" * 72,
            flush=True,
        )

        print(
            "Experiment:",
            config.experiment.name,
            flush=True,
        )

        print(
            "Accelerator:",
            accelerator,
            flush=True,
        )

        print(
            "Visible CUDA devices:",
            torch.cuda.device_count(),
            flush=True,
        )

        print(
            "Trainer devices:",
            devices,
            flush=True,
        )

        print(
            "Strategy:",
            strategy,
            flush=True,
        )

        if accelerator == "gpu":

            print(
                "CUDA_VISIBLE_DEVICES:",
                os.environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    "<not set>",
                ),
                flush=True,
            )

    progress_enabled = bool(
        get_config_value(
            config,
            "progress.enable_progress_bar",
            True,
        )
    )

    return L.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,

        max_epochs=int(
            config.trainer.max_epochs
        ),

        precision=(
            str(
                config.trainer.precision
            )
            if accelerator == "gpu"
            else "32-true"
        ),

        gradient_clip_val=float(
            config.trainer
            .gradient_clip_val
        ),

        callbacks=callbacks,
        logger=logger,

        deterministic=bool(
            get_config_value(
                config,
                "trainer.deterministic",
                True,
            )
        ),

        log_every_n_steps=int(
            get_config_value(
                config,
                "trainer.log_every_n_steps",
                20,
            )
        ),

        enable_progress_bar=(
            progress_enabled
        ),

        num_sanity_val_steps=int(
            get_config_value(
                config,
                "trainer.num_sanity_val_steps",
                2,
            )
        ),

        check_val_every_n_epoch=int(
            get_config_value(
                config,
                "trainer.check_val_every_n_epoch",
                1,
            )
        ),
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def print_dataset_summary(
    train_rows: list[dict],
    validation_rows: list[dict],
    train_sampling_mode: str,
    val_sampling_mode: str,
) -> None:
    """Print data summary on rank zero."""

    if not is_local_rank_zero():
        return

    train_real, train_fake = (
        count_binary_rows(
            train_rows
        )
    )

    val_real, val_fake = (
        count_binary_rows(
            validation_rows
        )
    )

    print(
        "\n"
        + "=" * 72
        + "\nDATASET"
        + "\n"
        + "=" * 72,
        flush=True,
    )

    print(
        "Train sampling mode:",
        train_sampling_mode,
        flush=True,
    )

    print(
        "Training samples   :",
        len(train_rows),
        flush=True,
    )

    print(
        "  Real             :",
        train_real,
        flush=True,
    )

    print(
        "  Fake             :",
        train_fake,
        flush=True,
    )

    print(
        "Val sampling mode  :",
        val_sampling_mode,
        flush=True,
    )

    print(
        "Validation samples :",
        len(validation_rows),
        flush=True,
    )

    print(
        "  Real             :",
        val_real,
        flush=True,
    )

    print(
        "  Fake             :",
        val_fake,
        flush=True,
    )


# ----------------------------------------------------------------------
# Final checkpoint handling
# ----------------------------------------------------------------------


def finalize_best_checkpoint(
    config: DictConfig,
    output_root: Path,
    checkpoint_root: Path,
    checkpoint_callback: ModelCheckpoint,
) -> None:
    """Copy selected best checkpoint to canonical filename."""

    best_checkpoint_path = (
        checkpoint_callback
        .best_model_path
    )

    if not best_checkpoint_path:
        raise RuntimeError(
            "ModelCheckpoint did not return "
            "a best checkpoint path."
        )

    best_checkpoint = Path(
        best_checkpoint_path
    ).resolve()

    if not best_checkpoint.is_file():
        raise RuntimeError(
            "Best checkpoint does not exist: "
            f"{best_checkpoint}"
        )

    canonical_name = str(
        get_config_value(
            config,
            "checkpoint.canonical_name",
            "controlled_detector.ckpt",
        )
    )

    canonical_checkpoint = (
        checkpoint_root
        / canonical_name
    )

    # Avoid SameFileError if paths ever coincide.
    if (
        best_checkpoint
        != canonical_checkpoint.resolve()
    ):
        shutil.copy2(
            best_checkpoint,
            canonical_checkpoint,
        )

    (
        output_root
        / "best_checkpoint.txt"
    ).write_text(
        str(best_checkpoint) + "\n",
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 72
        + "\nTRAINING COMPLETE"
        + "\n"
        + "=" * 72,
        flush=True,
    )

    print(
        "\nBest checkpoint:",
        best_checkpoint,
        flush=True,
    )

    print(
        "\nCanonical checkpoint:",
        canonical_checkpoint,
        flush=True,
    )

    print(
        "\nLatest resume checkpoint:",
        checkpoint_root
        / "last.ckpt",
        flush=True,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train the controlled forensic detector "
            "for shortcut-reliance calibration."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to detector YAML config.",
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Optional OmegaConf overrides, e.g. "
            "trainer.max_epochs=2 "
            "resume.enabled=false"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    config = load_config(
        args.config,
        args.overrides,
    )

    seed = int(
        config.experiment.seed
    )

    L.seed_everything(
        seed,
        workers=True,
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    output_root = Path(
        config.paths.output_root
    ).expanduser().resolve()

    checkpoint_root = (
        output_root
        / "checkpoints"
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if is_local_rank_zero():

        OmegaConf.save(
            config=config,
            f=(
                output_root
                / "resolved_detector_config.yaml"
            ),
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    (
        train_rows,
        validation_rows,
        train_sampling_mode,
        val_sampling_mode,
    ) = build_rows(
        config,
        seed,
    )

    print_dataset_summary(
        train_rows,
        validation_rows,
        train_sampling_mode,
        val_sampling_mode,
    )

    train_loader, validation_loader = (
        build_dataloaders(
            config,
            train_rows,
            validation_rows,
            seed,
        )
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    module = build_module(
        config
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    callbacks, checkpoint_callback = (
        build_callbacks(
            config,
            checkpoint_root,
        )
    )

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------

    logger = build_logger(
        config,
        output_root,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------

    trainer = build_trainer(
        config,
        callbacks,
        logger,
    )

    if trainer.is_global_zero:

        print(
            "Output root:",
            output_root,
            flush=True,
        )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    resume_checkpoint = (
        resolve_resume_checkpoint(
            config,
            checkpoint_root,
        )
    )

    if trainer.is_global_zero:

        if resume_checkpoint is None:

            print(
                "\n"
                + "=" * 72
                + "\nSTARTING TRAINING FROM SCRATCH"
                + "\n"
                + "=" * 72,
                flush=True,
            )

        else:

            print(
                "\n"
                + "=" * 72
                + "\nRESUMING TRAINING"
                + "\n"
                + "=" * 72
                + f"\nCheckpoint: "
                f"{resume_checkpoint}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    trainer.fit(
        module,
        train_dataloaders=(
            train_loader
        ),
        val_dataloaders=(
            validation_loader
        ),
        ckpt_path=(
            resume_checkpoint
        ),
    )

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------

    if trainer.is_global_zero:

        finalize_best_checkpoint(
            config,
            output_root,
            checkpoint_root,
            checkpoint_callback,
        )


if __name__ == "__main__":
    main()