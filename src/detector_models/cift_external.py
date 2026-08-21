from __future__ import annotations

import os
import sys
import warnings
from typing import Any

import torch
from torch import nn


class CIFTExternal(nn.Module):
    """Frozen score-access bridge to the existing CIFT implementation.

    This intentionally does not duplicate CIFT training code inside RIFT-v2.
    The new RIFT paper treats detectors as already-trained and frozen. This
    bridge mirrors the loading/forward convention used by the old RIFT project.
    """

    def __init__(self, cfg):
        super().__init__()
        if str(getattr(cfg, "mode", "frozen")) != "frozen":
            raise ValueError(
                "CIFTExternal is intentionally frozen in RIFT-v2. Retrain CIFT in its own repository, "
                "then point cift_ckpt here. This keeps the RIFT audit scientifically score-access only."
            )
        self.cfg = cfg
        self.cift_root = str(cfg.cift_root)
        self.config_path = str(cfg.cift_config)
        self.ckpt_path = str(cfg.cift_ckpt)
        self.backbone = str(getattr(cfg, "backbone", "convnextv2_base"))
        self.first_stage_key = str(getattr(cfg, "first_stage_key", "source"))
        self.target_stage_key = str(getattr(cfg, "target_stage_key", "target"))
        self.control_key = str(getattr(cfg, "control_key", "hint"))
        self.label_key = str(getattr(cfg, "label_key", "label"))
        self.model: Any = None
        self._load()

    def _load(self):
        if not os.path.isdir(self.cift_root):
            raise FileNotFoundError(f"CIFT repo root not found: {self.cift_root}")
        if self.cift_root not in sys.path:
            sys.path.insert(0, self.cift_root)
        from cldm.model import create_model  # type: ignore

        cfg_path = self.config_path
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(self.cift_root, cfg_path)
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"CIFT config not found: {cfg_path}")
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"CIFT checkpoint not found: {self.ckpt_path}")

        model = create_model(cfg_path)
        bb = self.backbone.split(".")[0]
        if hasattr(model, "control_model") and hasattr(model.control_model, "define_feature_filter"):
            model.control_model.define_feature_filter(bb)

        raw = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)
        state.pop("cond_stage_model.transformer.text_model.embeddings.position_ids", None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(unexpected) > 50:
            warnings.warn(
                f"CIFT load: {len(missing)} missing / {len(unexpected)} unexpected keys. "
                "Check checkpoint/backbone/config compatibility.",
                RuntimeWarning,
            )
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        self.model = model

    def train(self, mode: bool = True):
        # A RIFT detector must stay frozen even if Lightning toggles module mode.
        super().train(False)
        if self.model is not None:
            self.model.eval()
        return self

    def _batch(self, x: torch.Tensor) -> dict[str, Any]:
        b = x.shape[0]
        return {
            self.control_key: x,
            "hint_ori": x,
            self.target_stage_key: x,
            self.first_stage_key: x,  # source-free deployment path: donor is not supplied
            self.label_key: torch.ones(b, device=x.device),
            "forgery_type": "swap",
            "txt": [""] * b,
        }

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = self._batch(x)
        source, target, c, _ = self.model.get_input(batch, self.model.first_stage_key)
        out = self.model(source, target, c, batch[self.label_key])
        loss_dict = out[1] if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], dict) else {}
        if "v/logits" in loss_dict:
            return loss_dict["v/logits"].detach().float().view(-1)
        if "v/probs" in loss_dict:
            p = loss_dict["v/probs"].detach().float().clamp(1e-6, 1.0 - 1e-6).view(-1)
            return torch.logit(p)
        raise RuntimeError(f"CIFT forward did not expose v/logits or v/probs. Keys={list(loss_dict)}")
