from __future__ import annotations

import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryROC


class BinaryEER(Metric):
    """
    Distributed-safe Equal Error Rate (EER) for binary classification.

    Expected inputs:
        scores:
            Fake-class probabilities in [0, 1].
            Shape can be [B] or anything flattenable.

        targets:
            Binary labels:
                0 = real
                1 = fake

    Why BinaryROC is used:
        BinaryROC already handles synchronization of predictions and
        targets correctly under Distributed Data Parallel (DDP).

        The previous implementation stored two dynamically concatenated
        tensors with dist_reduce_fx="cat", which could become inconsistent
        across ranks and cause CUDA index-out-of-bounds errors.
    """

    full_state_update = False
    is_differentiable = False
    higher_is_better = False

    def __init__(self):
        super().__init__()

        # thresholds=None stores predictions/targets and computes
        # an exact ROC curve at compute() time.
        #
        # TorchMetrics handles distributed synchronization internally.
        self.roc = BinaryROC(thresholds=None)

    def update(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Add one batch of predictions and labels.
        """

        scores = scores.detach().float().flatten()
        targets = targets.detach().long().flatten()

        if scores.numel() != targets.numel():
            raise ValueError(
                "BinaryEER received different numbers of predictions "
                f"and targets: scores={scores.numel()}, "
                f"targets={targets.numel()}"
            )

        if scores.numel() == 0:
            return

        # Defensive label check.
        invalid = (targets != 0) & (targets != 1)

        if torch.any(invalid):
            bad_values = torch.unique(targets[invalid]).detach().cpu().tolist()

            raise ValueError(
                "BinaryEER expects binary targets {0, 1}, "
                f"but found invalid values: {bad_values}"
            )

        self.roc.update(scores, targets)

    def compute(self) -> torch.Tensor:
        """
        Compute Equal Error Rate from the globally synchronized ROC curve.

        EER is approximated at the ROC operating point where:

            False Positive Rate ~= False Negative Rate

        and returned as:

            0.5 * (FPR + FNR)
        """

        fpr, tpr, _ = self.roc.compute()

        # Defensive consistency check.
        if fpr.numel() != tpr.numel():
            raise RuntimeError(
                "Invalid ROC state while computing EER: "
                f"fpr has {fpr.numel()} values, "
                f"tpr has {tpr.numel()} values."
            )

        if fpr.numel() == 0:
            return torch.tensor(
                float("nan"),
                device=fpr.device,
                dtype=torch.float32,
            )

        fpr = fpr.float()
        tpr = tpr.float()

        fnr = 1.0 - tpr

        difference = torch.abs(fpr - fnr)

        # Protect against any NaN/Inf values.
        finite = torch.isfinite(difference)

        if not finite.any():
            return torch.tensor(
                float("nan"),
                device=fpr.device,
                dtype=torch.float32,
            )

        difference = torch.where(
            finite,
            difference,
            torch.full_like(
                difference,
                float("inf"),
            ),
        )

        idx = torch.argmin(difference)

        eer = 0.5 * (
            fpr[idx] + fnr[idx]
        )

        return eer

    def reset(self) -> None:
        """
        Reset metric state between validation epochs.
        """

        super().reset()
        self.roc.reset()


class BinaryBrier(Metric):
    """
    Distributed-safe Brier score for binary classification.

    Brier score:

        mean((probability - target)^2)

    Lower is better.

    Expected:
        probability = fake probability in [0, 1]
        target      = 0 for real, 1 for fake
    """

    full_state_update = False
    is_differentiable = False
    higher_is_better = False

    def __init__(self):
        super().__init__()

        # Fixed-size scalar states are naturally safe under DDP.
        self.add_state(
            "sum_sq",
            default=torch.tensor(
                0.0,
                dtype=torch.float32,
            ),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "count",
            default=torch.tensor(
                0,
                dtype=torch.long,
            ),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Add one batch to the Brier-score accumulator.
        """

        probs = probs.detach().float().flatten()
        targets = targets.detach().float().flatten()

        if probs.numel() != targets.numel():
            raise ValueError(
                "BinaryBrier received different numbers of predictions "
                f"and targets: probs={probs.numel()}, "
                f"targets={targets.numel()}"
            )

        if probs.numel() == 0:
            return

        self.sum_sq += torch.sum(
            (probs - targets) ** 2
        )

        self.count += probs.numel()

    def compute(self) -> torch.Tensor:
        """
        Return the globally reduced mean Brier score.
        """

        if self.count == 0:
            return torch.tensor(
                float("nan"),
                device=self.sum_sq.device,
                dtype=torch.float32,
            )

        return self.sum_sq / self.count.float()