"""Smoke test for the `--spectrum-dims K` (per-dim-subset spectrum) feature.

Avoids the heavy PLINK cohort-cache setup by mocking the parent
`RankBHeritability.__init__` (only `nn.Module.__init__` + a few attrs the spectrum
math needs), so the REAL `RankBHeritabilitySpectrum.__init__` (incl. the
rank_weights/spectrum_dims compatibility guard) and the REAL slicing math run.

Run: python3 tests/test_spectrum_dims.py
"""
from unittest.mock import patch
import torch
import torch.nn as nn

from h2vae.rank_b_heritability import RankBHeritability
from h2vae.rank_b_spectrum import RankBHeritabilitySpectrum


def _fake_parent_init(self, bed, row_idx, C=None, y_target=None, hweights=None,
                      device="cpu", chunk_variants=4096, b_hutch=10, seed_hutch=0,
                      dtype=torch.float32):
    nn.Module.__init__(self)              # enable register_buffer
    self.device = torch.device(device)
    self.dtype = dtype
    self.mode = "mom"
    self.has_W = True
    self.m = 100                          # dummy variant count


def _make(spectrum_dims=None, rank_weights=None, spectrum_clamp=True, ridge=1e-4):
    with patch.object(RankBHeritability, "__init__", _fake_parent_init):
        return RankBHeritabilitySpectrum(
            bed=None, row_idx=[0, 1, 2], C=None,
            ridge=ridge, spectrum_clamp=spectrum_clamp,
            rank_weights=rank_weights, spectrum_dims=spectrum_dims,
        )


def _rand_G_P(d, seed=0):
    """A symmetric PSD G and a unit-diagonal correlation P_corr (d×d)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    G = (A @ A.T) / d
    B = torch.randn(d, d, generator=g)
    S = (B @ B.T) / d + torch.eye(d)
    dsd = torch.sqrt(torch.diagonal(S))
    P = S / (dsd[:, None] * dsd[None, :])      # unit diagonal correlation
    return G, P


def test_ctor_guard():
    # matched length: ok
    _make(spectrum_dims=32, rank_weights=torch.ones(32))
    # oversized (128 with K=32): ok (sliced to [:32])
    rw = torch.cat([torch.zeros(1), torch.ones(127)])
    _make(spectrum_dims=32, rank_weights=rw)
    # too short (16 < K=32): must raise
    raised = False
    try:
        _make(spectrum_dims=32, rank_weights=torch.ones(16))
    except ValueError:
        raised = True
    assert raised, "ctor should raise ValueError when rank_weights shorter than K"
    print("OK ctor guard: matched/oversized accepted, short raises")


def test_spectrum_length_and_finite():
    d, K = 128, 32
    G, P = _rand_G_P(d)
    full = _make(spectrum_dims=None)
    lam_full = full._spectrum(G, P)
    assert lam_full.shape[0] == d, lam_full.shape
    assert torch.isfinite(lam_full).all()

    sub = _make(spectrum_dims=K)
    lam_sub = sub._spectrum(G, P)
    assert lam_sub.shape[0] == K, lam_sub.shape
    assert torch.isfinite(lam_sub).all()

    # the K-dim spectrum must equal the spectrum of the explicit top-left block
    ref = _make(spectrum_dims=None)._spectrum(G[:K, :K].contiguous(), P[:K, :K].contiguous())
    assert torch.allclose(lam_sub, ref, atol=1e-5), (lam_sub - ref).abs().max()
    print(f"OK spectrum length: full={d}, sub={K}; sub == top-left-block spectrum")


def test_loss_rank_weight_compat():
    d, K = 128, 32
    G, P = _rand_G_P(d, seed=1)

    # uniform K-length weights ⇒ loss == -sum(K eigenvalues, clamped)
    sub_u = _make(spectrum_dims=K, rank_weights=torch.ones(K))
    lam = sub_u._spectrum(G, P).clamp_min(0)
    loss_u = sub_u._spectrum_loss(G, P)
    assert torch.isfinite(loss_u)
    assert torch.allclose(loss_u, -lam.sum(), atol=1e-5), (loss_u, -lam.sum())

    # oversized 128-length weights with K=32 ⇒ uses [:K]; finite, matches uniform here
    rw128 = torch.ones(d)
    sub_o = _make(spectrum_dims=K, rank_weights=rw128)
    loss_o = sub_o._spectrum_loss(G, P)
    assert torch.isfinite(loss_o)
    assert torch.allclose(loss_o, loss_u, atol=1e-5)

    # all_but_top within K: zero the top eigenvalue's contribution
    abt = torch.ones(K); abt[0] = 0
    sub_abt = _make(spectrum_dims=K, rank_weights=abt)
    loss_abt = sub_abt._spectrum_loss(G, P)
    expected = -(lam[1:K] * abt[1:]).sum()
    assert torch.allclose(loss_abt, expected, atol=1e-5), (loss_abt, expected)

    # no weights ⇒ -sum
    sub_none = _make(spectrum_dims=K, rank_weights=None)
    assert torch.allclose(sub_none._spectrum_loss(G, P), -lam.sum(), atol=1e-5)
    print("OK loss/rank-weight compat: uniform, oversized-sliced, all_but_top, none")


if __name__ == "__main__":
    test_ctor_guard()
    test_spectrum_length_and_finite()
    test_loss_rank_weight_compat()
    print("\nALL SPECTRUM-DIMS SMOKE TESTS PASSED")
