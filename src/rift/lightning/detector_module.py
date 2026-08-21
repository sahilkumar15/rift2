from __future__ import annotations

import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryROC,
)

from rift.metrics.binary import BinaryBrier

from rift.metrics.binary import BinaryBrier, BinaryEER
from rift.models.factory import build_model


class DetectorLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model, self.trainable_detector = build_model(cfg.model)
        self.save_hyperparameters(ignore=["cfg"])

        self.val_auc = BinaryAUROC()
        self.val_ap = BinaryAveragePrecision()
        self.val_acc = BinaryAccuracy(threshold=0.5)

        # Distributed-safe ROC.
        # We derive EER from this global ROC curve.
        self.val_roc = BinaryROC(thresholds=None)

        self.val_ece = BinaryCalibrationError(
            n_bins=15,
            norm="l1",
        )
        self.val_brier = BinaryBrier()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_smoothing = float(getattr(self.cfg.loss, "label_smoothing", 0.0)) if hasattr(self.cfg, "loss") else 0.0
        y = labels.float()
        if label_smoothing > 0:
            y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
        pos_weight_cfg = getattr(self.cfg.loss, "pos_weight", None) if hasattr(self.cfg, "loss") else None
        pos_weight = None
        if pos_weight_cfg not in (None, "null"):
            pos_weight = torch.tensor(float(pos_weight_cfg), device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

    def training_step(self, batch, batch_idx):
        if not self.trainable_detector:
            raise RuntimeError("This detector is frozen/evaluation-only and cannot be trained by RIFT-v2.")
        logits = self(batch["image"])
        labels = batch["label"].float().view_as(logits)
        loss = self._loss(logits, labels)
        probs = torch.sigmoid(logits.detach())
        acc = ((probs >= 0.5).float() == labels).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=labels.numel())
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=labels.numel())
        return loss

    def on_train_epoch_start(self):
        dm = self.trainer.datamodule
        if dm is not None and hasattr(dm, "set_epoch"):
            dm.set_epoch(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        logits = self(batch["image"])
        labels = batch["label"].float().view_as(logits)
        loss = self._loss(logits, labels)
        probs = torch.sigmoid(logits)
        y_int = labels.long()

        self.val_auc.update(probs, y_int)
        self.val_ap.update(probs, y_int)
        self.val_acc.update(probs, y_int)

        # ROC keeps predictions/targets using TorchMetrics'
        # distributed-safe implementation.
        self.val_roc.update(probs, y_int)

        self.val_ece.update(probs, y_int)
        self.val_brier.update(probs, y_int)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=labels.numel())

    def on_validation_epoch_end(self):
        # ---------------------------------------------------------
        # EER from globally synchronized ROC
        # ---------------------------------------------------------
        fpr, tpr, _ = self.val_roc.compute()

        if fpr.numel() == 0 or tpr.numel() == 0:
            val_eer = torch.tensor(
                float("nan"),
                device=self.device,
            )
        else:
            fnr = 1.0 - tpr

            diff = torch.abs(fpr - fnr)

            if not torch.isfinite(diff).any():
                val_eer = torch.tensor(
                    float("nan"),
                    device=self.device,
                )
            else:
                # Protect argmin if a non-finite value appears.
                diff = torch.nan_to_num(
                    diff,
                    nan=float("inf"),
                    posinf=float("inf"),
                    neginf=float("inf"),
                )

                idx = torch.argmin(diff)

                val_eer = 0.5 * (
                    fpr[idx] + fnr[idx]
                )

        # ---------------------------------------------------------
        # Other validation metrics
        # Their compute() calls also synchronize correctly in DDP.
        # ---------------------------------------------------------
        vals = {
            "val/auc": self.val_auc.compute(),
            "val/ap": self.val_ap.compute(),
            "val/acc": self.val_acc.compute(),
            "val/eer": val_eer,
            "val/ece": self.val_ece.compute(),
            "val/brier": self.val_brier.compute(),
        }

        self.log_dict(
            vals,
            prog_bar=True,
            sync_dist=True,
        )

        # ---------------------------------------------------------
        # Reset every metric for the next validation epoch.
        # ---------------------------------------------------------
        for metric in (
            self.val_auc,
            self.val_ap,
            self.val_acc,
            self.val_roc,
            self.val_ece,
            self.val_brier,
        ):
            metric.reset()

    def configure_optimizers(self):
        if not self.trainable_detector:
            return None
        opt_cfg = self.cfg.optimizer
        if str(opt_cfg.name).lower() != "adamw":
            raise ValueError("Only optimizer.name=adamw is implemented in the clean baseline")
        optimizer = AdamW(self.parameters(), lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay))

        sch_cfg = self.cfg.scheduler
        if str(sch_cfg.name).lower() != "cosine":
            return optimizer

        max_epochs = int(self.cfg.trainer.max_epochs)
        warmup = max(0, int(getattr(sch_cfg, "warmup_epochs", 0)))
        min_lr = float(getattr(sch_cfg, "min_lr", 0.0))
        base_lr = float(opt_cfg.lr)
        eta_min = min_lr
        if warmup <= 0:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs), eta_min=eta_min)
        else:
            start_factor = max(min_lr / base_lr, 1e-3)
            warm = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup)
            cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup), eta_min=eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
