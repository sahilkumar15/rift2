import torch

from forensic_audit.fss import compute_fss, harmonic_fss, robust_score_scale


def test_harmonic_limits():
    assert torch.isclose(harmonic_fss(torch.tensor(0.0), torch.tensor(0.0)), torch.tensor(0.0))
    assert torch.isclose(harmonic_fss(torch.tensor(1.0), torch.tensor(1.0)), torch.tensor(0.0))
    assert harmonic_fss(torch.tensor(1.0), torch.tensor(0.0)) > 0.999


def test_robust_scale():
    s = torch.arange(100, dtype=torch.float32)
    scale = robust_score_scale(s)
    assert 80 < scale < 100


def test_fss_score_access_contract():
    # Detector score depends on the top-left region mean.
    def score(x):
        return x[:, :, :2, :2].mean(dim=(1, 2, 3)) * 4.0

    real = torch.zeros(4, 3, 4, 4)
    fake = real.clone()
    fake[:, :, :2, :2] = 1.0
    mask = torch.zeros(4, 1, 4, 4)
    mask[:, :, :2, :2] = 1.0

    def zero_region(x, m):
        return x * (1 - m)

    nuis = [("identity", lambda x: x.clone())]
    out = compute_fss(score, real, fake, mask, intervention=zero_region, nuisances=nuis, score_scale=4.0)
    assert out.manipulation_reliance > 0.99
    assert out.nuisance_instability < 1e-6
    assert out.fss > 0.99
