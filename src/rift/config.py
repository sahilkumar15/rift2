from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(str(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def to_plain_dict(cfg: Any) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def active_domain_weights(cfg: DictConfig) -> dict[str, float]:
    if str(cfg.dataset.name) != "mixed":
        return {str(cfg.dataset.name): 1.0}
    domains = list(cfg.dataset.mixed.domains)
    if not domains:
        raise ValueError("dataset.mixed.domains is empty")
    raw = {str(d.name): float(d.weight) for d in domains if float(d.weight) > 0}
    if not raw:
        raise ValueError("All mixed-domain weights are non-positive")
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}
