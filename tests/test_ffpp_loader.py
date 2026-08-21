import json
from pathlib import Path

from PIL import Image
from omegaconf import OmegaConf

from rift.data.ffpp_relation import FFPPRelationDataset
from rift.data.transforms import build_transform


def _save(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (value, value, value)).save(path)


def test_ffpp_relation_scan(tmp_path):
    root = tmp_path / "ffpp"
    _save(root / "original_sequences/youtube/c23/images/000/000_0001.png", 20)
    _save(root / "original_sequences/youtube/c23/images/001/001_0001.png", 30)
    _save(root / "manipulated_sequences/Deepfakes/c23/images/000_001/000_001_0001.png", 220)
    (root / "splits").mkdir(parents=True)
    (root / "splits/train.json").write_text(json.dumps([["000", "001"]]))

    cfg = OmegaConf.create({
        "data_root": str(root), "compressions": "c23", "num_frames": 1,
        "methods": ["youtube", "Deepfakes"], "balance": False,
        "use_splits": True, "strict_splits": True, "splits_dirname": "splits",
    })
    tcfg = OmegaConf.create({"image_size": 16, "mean": [0.5]*3, "std": [0.5]*3, "aug": {"enable": False}})
    ds = FFPPRelationDataset(cfg, build_transform(tcfg, train=False), split="train")
    assert len(ds) == 3
    fake = next(ds[i] for i in range(len(ds)) if int(ds[i]["label"].item()) == 1)
    assert bool(fake["relation_valid"].item())
    assert "000_0001.png" in fake["donor_path"]
