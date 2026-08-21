from __future__ import annotations

import torch


class TemperatureScaledScore:
    """Optional score calibration wrapper. RIFT operates on logits, not raw probabilities."""
    def __init__(self, model, temperature: float = 1.0):
        self.model = model
        self.temperature = max(float(temperature), 1e-6)

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).float().flatten() / self.temperature
