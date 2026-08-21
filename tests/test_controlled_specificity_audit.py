from pathlib import Path

import torch

from controlled_forensic_audit.shortcut import PlantedShortcutSpec
from controlled_forensic_audit.specificity_audit.regions import build_evidence_masks
from forensic_audit.fss import compute_fss_from_scores


def test_flat_source_layout_has_no_rift_namespace():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src" / "rift").exists()
    assert (root / "src" / "controlled_forensic_audit").is_dir()
    assert (root / "src" / "forensic_audit").is_dir()
    assert (root / "src" / "detector_training").is_dir()


def test_control_regions_preserve_area_and_avoid_forbidden_pixels():
    pristine = torch.zeros(3, 64, 64)
    pristine[:, 24:32, 24:32] = 0.5

    gt = torch.zeros(1, 64, 64, dtype=torch.bool)
    gt[:, 24:32, 24:32] = True

    spec = PlantedShortcutSpec(
        size_px=8,
        top_px=4,
        left_px=4,
        tile_px=2,
    )

    regions = build_evidence_masks(
        pristine=pristine,
        gt_mask=gt,
        sample_id="Deepfakes/000_001/0001",
        shortcut_spec=spec,
        seed=3407,
        candidate_stride=4,
    )

    target_area = int(gt.sum())
    assert int(regions.gt_manipulation.sum()) == target_area
    assert int(regions.matched_background.sum()) == target_area
    assert int(regions.random_region.sum()) == target_area

    forbidden = regions.gt_manipulation.bool() | regions.planted_shortcut.bool()
    assert not bool((regions.matched_background.bool() & forbidden).any())
    assert not bool((regions.random_region.bool() & forbidden).any())


def test_score_only_fss_shared_nuisance_contract():
    fake = torch.tensor([2.0, 2.0])
    pristine = torch.tensor([0.0, 0.0])
    fake_removed = torch.tensor([0.5, 0.5])
    pristine_removed = torch.tensor([0.0, 0.0])

    nuisances = {
        "identity": (
            fake.clone(),
            pristine.clone(),
            fake_removed.clone(),
            pristine_removed.clone(),
        )
    }

    out = compute_fss_from_scores(
        fake_score=fake,
        pristine_score=pristine,
        fake_removed_score=fake_removed,
        pristine_removed_score=pristine_removed,
        nuisance_contributions=nuisances,
        score_scale=2.0,
    )

    assert torch.all(out.manipulation_reliance > 0.7)
    assert torch.all(out.nuisance_instability < 1e-7)
    assert torch.all(out.fss > 0.8)
