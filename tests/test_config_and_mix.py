from pathlib import Path

from project_core.config import active_domain_weights, load_config


def test_normalized_weights():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/train_detector_mixed.yaml")
    w = active_domain_weights(cfg)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert w["ffpp_rela"] > w["dfd"] > w["celeb_df"] > w["wild_deepfake"] > w["diffswap"]
