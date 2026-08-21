from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import BinaryAUROC
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    convnext_tiny,
)

from .shortcut import (
    PlantedShortcutSpec,
    apply_planted_shortcut,
)


class ControlledForensicDetectorModule(
    L.LightningModule
):
    """
    Detector deliberately trained to contain both:

      1. genuine image-content sensitivity; and
      2. reliance on a controlled planted shortcut.

    It is used only to validate the forensic-specificity
    auditing instrument.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        clean_supervision_weight: float = 0.5,
        shortcut_size_px: int = 32,
        shortcut_top_px: int = 8,
        shortcut_left_px: int = 8,
        shortcut_tile_px: int = 4,
        shortcut_low_value: float = 0.05,
        shortcut_high_value: float = 0.95,
    ) -> None:

        super().__init__()

        self.save_hyperparameters()

        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained
            else None
        )

        self.detector = convnext_tiny(
            weights=weights
        )

        in_features = (
            self.detector
            .classifier[-1]
            .in_features
        )

        self.detector.classifier[-1] = (
            nn.Linear(
                in_features,
                1,
            )
        )

        self.learning_rate = float(
            learning_rate
        )

        self.weight_decay = float(
            weight_decay
        )

        self.clean_supervision_weight = float(
            clean_supervision_weight
        )

        self.shortcut_spec = (
            PlantedShortcutSpec(
                size_px=shortcut_size_px,
                top_px=shortcut_top_px,
                left_px=shortcut_left_px,
                tile_px=shortcut_tile_px,
                low_value=shortcut_low_value,
                high_value=shortcut_high_value,
            )
        )

        self.register_buffer(
            "normalization_mean",
            torch.tensor(
                [
                    0.485,
                    0.456,
                    0.406,
                ]
            ).view(
                1,
                3,
                1,
                1,
            ),
        )

        self.register_buffer(
            "normalization_std",
            torch.tensor(
                [
                    0.229,
                    0.224,
                    0.225,
                ]
            ).view(
                1,
                3,
                1,
                1,
            ),
        )

        self.validation_auc_without_shortcut = (
            BinaryAUROC()
        )

        self.validation_auc_shortcut_aligned = (
            BinaryAUROC()
        )

    def normalize(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:

        return (
            images
            - self.normalization_mean
        ) / self.normalization_std

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:

        normalized = self.normalize(
            images
        )

        return (
            self.detector(
                normalized
            )
            .view(-1)
        )

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> torch.Tensor:

        clean_images = batch[
            "image"
        ]

        labels = batch[
            "label"
        ].float()

        shortcut_present = batch[
            "shortcut_present"
        ].bool()

        shortcut_images = (
            apply_planted_shortcut(
                clean_images,
                self.shortcut_spec,
            )
        )

        selector = (
            shortcut_present
            .view(-1, 1, 1, 1)
        )

        shortcut_correlated_images = (
            torch.where(
                selector,
                shortcut_images,
                clean_images,
            )
        )

        correlated_logits = self(
            shortcut_correlated_images
        )

        clean_logits = self(
            clean_images
        )

        correlated_loss = (
            F.binary_cross_entropy_with_logits(
                correlated_logits,
                labels,
            )
        )

        clean_loss = (
            F.binary_cross_entropy_with_logits(
                clean_logits,
                labels,
            )
        )

        loss = (
            correlated_loss
            + self.clean_supervision_weight
            * clean_loss
        )

        correlated_accuracy = (
            (
                (
                    torch.sigmoid(
                        correlated_logits
                    )
                    >= 0.5
                )
                == labels.bool()
            )
            .float()
            .mean()
        )

        batch_size = int(labels.numel())

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        self.log(
            "train_correlated_accuracy",
            correlated_accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> None:

        images = batch[
            "image"
        ]

        labels = (
            batch["label"]
            .long()
        )

        images_with_shortcut = (
            apply_planted_shortcut(
                images,
                self.shortcut_spec,
            )
        )

        logits_without_shortcut = self(
            images
        )

        logits_with_shortcut = self(
            images_with_shortcut
        )

        probabilities_without_shortcut = (
            torch.sigmoid(
                logits_without_shortcut
            )
        )

        probabilities_with_shortcut = (
            torch.sigmoid(
                logits_with_shortcut
            )
        )

        # Same condition for both classes:
        # actual forensic-content sensitivity.
        self.validation_auc_without_shortcut.update(
            probabilities_without_shortcut,
            labels,
        )

        # Counterfactual shortcut-aligned condition:
        #
        #   genuine -> shortcut absent
        #   fake    -> shortcut present
        #
        aligned_probabilities = torch.where(
            labels.bool(),
            probabilities_with_shortcut,
            probabilities_without_shortcut,
        )

        self.validation_auc_shortcut_aligned.update(
            aligned_probabilities,
            labels,
        )

    def on_validation_epoch_end(
        self,
    ) -> None:

        auc_without_shortcut = (
            self
            .validation_auc_without_shortcut
            .compute()
        )

        auc_shortcut_aligned = (
            self
            .validation_auc_shortcut_aligned
            .compute()
        )

        control_score = (
            0.5
            * (
                auc_without_shortcut
                + auc_shortcut_aligned
            )
        )

        self.log(
            "val_auc_without_shortcut",
            auc_without_shortcut,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )

        self.log(
            "val_auc_shortcut_aligned",
            auc_shortcut_aligned,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )

        self.log(
            "val_control_score",
            control_score,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )

        self.validation_auc_without_shortcut.reset()
        self.validation_auc_shortcut_aligned.reset()

    def configure_optimizers(
        self,
    ):

        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )