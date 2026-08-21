from __future__ import annotations

import os

# ------------------------------------------------------------------
# Hugging Face cache compatibility.
# Must happen before importing packages that may import transformers.
# ------------------------------------------------------------------
_legacy_cache = os.environ.pop("TRANSFORMERS_CACHE", None)

if _legacy_cache:
    os.environ.setdefault("HF_HOME", _legacy_cache)


import lightning.pytorch as pl
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from rift.lightning.detector_module import DetectorLightningModule


# ==================================================================
# Toy dataset
# ==================================================================

class Toy(Dataset):
    """
    Tiny deterministic binary-classification dataset.

    Real:
        label = 0
        all-zero image

    Fake:
        label = 1
        top-left region is set to 1

    This gives the Lightning smoke test a simple learnable pattern
    without depending on the real FF++ dataset.
    """

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        image = torch.zeros(
            3,
            16,
            16,
            dtype=torch.float32,
        )

        label = float(index % 2)

        if label == 1.0:
            image[:, :8, :8] = 1.0

        return {
            "image": image,

            # Relation-aware modules expect donor to exist.
            "donor": image.clone(),

            "label": torch.tensor(
                label,
                dtype=torch.float32,
            ),

            "relation_valid": torch.tensor(
                True,
                dtype=torch.bool,
            ),

            "domain": "toy",

            "path": str(index),

            "donor_path": str(index),

            "sample_id": str(index),

            "forgery_type": (
                "genuine"
                if label == 0.0
                else "swap"
            ),
        }


# ==================================================================
# Toy Lightning DataModule
# ==================================================================

class DM(pl.LightningDataModule):

    def train_dataloader(self) -> DataLoader:
        # num_workers=0 is intentional for a tiny unit test.
        # Starting subprocess workers would cost more than loading
        # these eight synthetic samples directly.
        return DataLoader(
            Toy(),
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Toy(),
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

    def set_epoch(self, epoch: int) -> None:
        # RIFTDataModule exposes this method, so the toy module
        # provides the same minimal interface.
        del epoch


# ==================================================================
# Lightning smoke test
# ==================================================================

@pytest.mark.filterwarnings(
    "ignore:The `srun` command is available on your system but is not used.*"
)
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers.*"
)
@pytest.mark.filterwarnings(
    "ignore:The 'val_dataloader' does not have many workers.*"
)
def test_one_epoch(tmp_path):
    """
    Verify that DetectorLightningModule can complete:

        forward
        loss
        backward
        optimizer step
        validation
        metrics

    for one tiny CPU epoch.
    """

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "tiny_cnn",
            },

            "loss": {
                "pos_weight": None,
                "label_smoothing": 0.0,
            },

            "optimizer": {
                "name": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.0,
            },

            "scheduler": {
                "name": "cosine",
                "warmup_epochs": 0,
                "min_lr": 1e-5,
            },

            "trainer": {
                "max_epochs": 1,
            },
        }
    )

    model = DetectorLightningModule(
        cfg
    )

    trainer = pl.Trainer(
        # ------------------------------------------------------
        # Explicit CPU test.
        # This prevents Lightning from treating the available
        # A100 GPUs as accidentally unused.
        # ------------------------------------------------------
        accelerator="cpu",
        devices=1,

        # ------------------------------------------------------
        # Keep the unit test extremely short.
        # ------------------------------------------------------
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=2,

        # ------------------------------------------------------
        # Disable things unnecessary for a unit test.
        # ------------------------------------------------------
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,

        # Full validation happens after the training batch, so
        # the preliminary Lightning sanity-validation pass is
        # unnecessary here.
        num_sanity_val_steps=0,

        # Put temporary Lightning artifacts inside pytest's
        # temporary directory.
        default_root_dir=str(tmp_path),
    )

    trainer.fit(
        model,
        datamodule=DM(),
    )

    # ----------------------------------------------------------
    # Basic completion assertions
    # ----------------------------------------------------------

    assert trainer.current_epoch == 1

    assert trainer.global_step > 0