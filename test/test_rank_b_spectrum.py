"""Tests for the rank-B heritability-spectrum module.

Verifies that ``RankBHeritabilitySpectrum`` (rank-B, incremental) reproduces:

  * the dense reference ``h2vae.heritability.gcov_spectrum`` (G, P_corr, spectrum),
  * the SCORE-OVERLAP ``gc()`` estimator exactly for every latent pair (G ≡ gc),
  * the weighted-spectrum objective (uniform, top-k via one-hot, and clamp),
  * the gradient through an encoder (matches the dense reference),
  * stability across many rank-B minibatch updates.

The Hutchinson ``tr_pkp2`` is patched to its exact value (W=[1|C]) so the
comparisons are exact, mirroring ``test_rank_b_heritability.py``.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "test"))

from h2vae.rank_b_spectrum import RankBHeritabilitySpectrum          # noqa: E402
from h2vae.heritability import gcov_spectrum, gc                     # noqa: E402
# make_fixture (ternary G → standardised X + synthetic PLINK trio) is defined in
# the rank-B heritability test module; reuse it for the identical convention.
from test_rank_b_heritability import make_fixture                    # noqa: E402


RIDGE = 1e-4


def patch_tr_pkp2_spectrum(her: RankBHeritabilitySpectrum, X: torch.Tensor,
                           C: torch.Tensor | None) -> None:
    """Replace the Hutchinson tr_pkp2 with the exact value (W = [1 | C])."""
    n = X.shape[0]
    K = (X @ X.T) / X.shape[1]
    ones = torch.ones((n, 1), dtype=X.dtype)
    W = ones if C is None else torch.hstack((ones, C))
    P = torch.eye(n, dtype=X.dtype) - W @ torch.linalg.inv(W.T @ W) @ W.T
    PKP = P @ K @ P
    her.tr_pkp2 = torch.tensor(float(torch.trace(PKP @ PKP)),
                               device=her.device, dtype=her.dtype)


def _rankb_matrices(her: RankBHeritabilitySpectrum, Z: torch.Tensor):
    """G, P_corr from the rank-B module on the full cohort Z (post-rebuild)."""
    her.rebuild(Z)
    mu = Z.mean(dim=0, keepdim=True)
    sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
    return her._genetic_cov_matrix(her.u_raw, her.w_raw, Z, mu, sd)


# ---------------------------------------------------------------------------
# Match the dense reference
# ---------------------------------------------------------------------------

def test_spectrum_matches_dense_no_C() -> None:
    n, m, zdim = 80, 50, 6
    _, X, _, bed, row_idx = make_fixture(n, m, seed=1)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2))

    her = RankBHeritabilitySpectrum(bed, row_idx, C=None, ridge=RIDGE,
                                    dtype=torch.float64)
    patch_tr_pkp2_spectrum(her, X, C=None)
    G_rb, P_rb = _rankb_matrices(her, Z)
    spec_rb = her._spectrum(G_rb, P_rb)

    G_d, P_d, spec_d = gcov_spectrum(X, C=None, ridge=RIDGE)(Z)
    assert torch.allclose(G_rb, G_d, atol=1e-9), (G_rb - G_d).abs().max()
    assert torch.allclose(P_rb, P_d, atol=1e-9)
    assert torch.allclose(spec_rb, spec_d, atol=1e-8)
    print(f"  no-C dense match  | spectrum = {spec_rb.numpy()}")


def test_spectrum_matches_dense_with_C() -> None:
    n, m, zdim, c = 100, 60, 5, 4
    _, X, _, bed, row_idx = make_fixture(n, m, seed=10)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(20))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(30))

    her = RankBHeritabilitySpectrum(bed, row_idx, C=C, ridge=RIDGE,
                                    dtype=torch.float64)
    patch_tr_pkp2_spectrum(her, X, C=C)
    G_rb, P_rb = _rankb_matrices(her, Z)
    spec_rb = her._spectrum(G_rb, P_rb)

    G_d, P_d, spec_d = gcov_spectrum(X, C=C, ridge=RIDGE)(Z)
    assert torch.allclose(G_rb, G_d, atol=1e-8), (G_rb - G_d).abs().max()
    assert torch.allclose(P_rb, P_d, atol=1e-8)
    assert torch.allclose(spec_rb, spec_d, atol=1e-7)
    print(f"  with-C dense match  | spectrum = {spec_rb.numpy()}")


# ---------------------------------------------------------------------------
# G ≡ gc() applied to every latent pair (the load-bearing identity, M1)
# ---------------------------------------------------------------------------

def test_G_matches_gc_pairwise_no_C() -> None:
    n, m, zdim = 80, 50, 5
    _, X, _, bed, row_idx = make_fixture(n, m, seed=40)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(50))

    her = RankBHeritabilitySpectrum(bed, row_idx, C=None, ridge=RIDGE,
                                    dtype=torch.float64)
    patch_tr_pkp2_spectrum(her, X, C=None)
    G_rb, _ = _rankb_matrices(her, Z)

    # Column j of G is gc(X, z_j, C)(Z) — γ̂(z_i, z_j) for all i.
    G_gc = torch.zeros(zdim, zdim, dtype=torch.float64)
    for j in range(zdim):
        gc_fn = gc(X, Z[:, j:j + 1], kinship=False, C=None,
                   device=torch.device("cpu"))
        G_gc[:, j] = gc_fn(Z)
    assert torch.allclose(G_rb, G_gc, atol=1e-9), (G_rb - G_gc).abs().max()
    print(f"  G ≡ gc() no-C  | max diff = {(G_rb - G_gc).abs().max().item():.2e}")


def test_G_matches_gc_pairwise_with_C() -> None:
    n, m, zdim, c = 90, 55, 4, 4
    _, X, _, bed, row_idx = make_fixture(n, m, seed=60)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(70))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(80))

    her = RankBHeritabilitySpectrum(bed, row_idx, C=C, ridge=RIDGE,
                                    dtype=torch.float64)
    patch_tr_pkp2_spectrum(her, X, C=C)
    G_rb, _ = _rankb_matrices(her, Z)

    G_gc = torch.zeros(zdim, zdim, dtype=torch.float64)
    for j in range(zdim):
        gc_fn = gc(X, Z[:, j:j + 1], kinship=False, C=C,
                   device=torch.device("cpu"))
        G_gc[:, j] = gc_fn(Z)
    assert torch.allclose(G_rb, G_gc, atol=1e-8), (G_rb - G_gc).abs().max()
    print(f"  G ≡ gc() with-C  | max diff = {(G_rb - G_gc).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
# Objective: uniform = trace, top-k via one-hot, clamp
# ---------------------------------------------------------------------------

def test_objective_uniform_topk_clamp() -> None:
    n, m, zdim, c = 90, 55, 6, 3
    _, X, _, bed, row_idx = make_fixture(n, m, seed=90)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(100))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(110))

    her = RankBHeritabilitySpectrum(bed, row_idx, C=C, ridge=RIDGE,
                                    dtype=torch.float64)
    patch_tr_pkp2_spectrum(her, X, C=C)
    G, P = _rankb_matrices(her, Z)
    lam = her._spectrum(G, P)

    # uniform weights ⇒ -sum(spectrum) = -tr(P_reg^{-1} G)
    loss_uniform = her._spectrum_loss(G, P)
    P_reg = P + RIDGE * torch.eye(zdim, dtype=torch.float64)
    trace_val = torch.trace(torch.linalg.solve(P_reg, G))
    assert torch.allclose(loss_uniform, -lam.sum(), atol=1e-10)
    assert torch.allclose(-loss_uniform, trace_val, atol=1e-7), (
        f"{-loss_uniform.item()} vs tr={trace_val.item()}"
    )

    # top-k via one-hot rank_weights
    k = 3
    rw = torch.zeros(zdim, dtype=torch.float64)
    rw[:k] = 1.0
    her.rank_weights = rw.to(her.device)
    loss_topk = her._spectrum_loss(G, P)
    assert torch.allclose(loss_topk, -lam[:k].sum(), atol=1e-10)

    # clamp: relu the spectrum
    her.rank_weights = None
    her.spectrum_clamp = True
    loss_clamp = her._spectrum_loss(G, P)
    assert torch.allclose(loss_clamp, -lam.clamp_min(0).sum(), atol=1e-10)
    print(f"  objective uniform/topk/clamp OK  | Σλ = {lam.sum().item():.4f}")


# ---------------------------------------------------------------------------
# Combined spectrum + marginal objective (two lambdas)
# ---------------------------------------------------------------------------

def test_combined_spectrum_marginal() -> None:
    n, m, zdim, c = 90, 55, 5, 4
    _, X, _, bed, row_idx = make_fixture(n, m, seed=120)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(130))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(140))
    ls, lm = 0.7, 0.3

    her = RankBHeritabilitySpectrum(bed, row_idx, C=C, ridge=RIDGE,
                                    spectrum_weight=ls, marginal_weight=lm,
                                    dtype=torch.float64)
    patch_tr_pkp2_spectrum(her, X, C=C)
    her.rebuild(Z)
    mu = Z.mean(0, keepdim=True)
    sd = Z.std(0, unbiased=True, keepdim=True).clamp_min(1e-8)
    G, P = her._genetic_cov_matrix(her.u_raw, her.w_raw, Z, mu, sd)
    spec_loss = her._spectrum_loss(G, P)
    marg_loss = -her._mom_h2(her.u_raw, her.w_raw, Z, mu, sd).sum()
    expected = ls * spec_loss + lm * marg_loss

    # zero-delta update reproduces the combined loss
    idxs = torch.arange(n)
    loss = her.update_and_loss(Z, idxs)
    assert torch.allclose(loss, expected, atol=1e-9), (loss, expected)

    # marginal_weight=0 recovers the pure spectrum loss
    her0 = RankBHeritabilitySpectrum(bed, row_idx, C=C, ridge=RIDGE,
                                     dtype=torch.float64)
    patch_tr_pkp2_spectrum(her0, X, C=C)
    her0.rebuild(Z)
    G0, P0 = her0._genetic_cov_matrix(her0.u_raw, her0.w_raw, Z, mu, sd)
    assert torch.allclose(her0.update_and_loss(Z, torch.arange(n)),
                          her0._spectrum_loss(G0, P0), atol=1e-9)
    print(f"  combined loss = {ls}·spec + {lm}·marg OK  "
          f"(spec={spec_loss.item():.4f}, marg={marg_loss.item():.4f})")


# ---------------------------------------------------------------------------
# Held-out spectrum: train eigenvectors evaluated on val data
# ---------------------------------------------------------------------------

def test_heldout_directions() -> None:
    n, m, zdim, c = 90, 55, 5, 4
    _, X, _, bed, row_idx = make_fixture(n, m, seed=150)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(160))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(170))

    her = RankBHeritabilitySpectrum(bed, row_idx, C=C, ridge=RIDGE,
                                    dtype=torch.float64)
    patch_tr_pkp2_spectrum(her, X, C=C)

    # (1) Scoring a cohort's OWN eigenvectors recovers its spectrum exactly.
    lam, W = her.eig_decompose(Z)
    h2_self = her.heritability_of_directions(Z, W)
    assert torch.allclose(h2_self, lam, atol=1e-8), (h2_self - lam).abs().max()

    # (2) eigenvectors are P_reg-orthonormal (Wᵀ P_reg W = I).
    her.rebuild(Z)
    mu = Z.mean(0, keepdim=True)
    sd = Z.std(0, unbiased=True, keepdim=True).clamp_min(1e-8)
    G, P = her._genetic_cov_matrix(her.u_raw, her.w_raw, Z, mu, sd)
    P_reg = P + RIDGE * torch.eye(zdim, dtype=torch.float64)
    assert torch.allclose(W.T @ P_reg @ W, torch.eye(zdim, dtype=torch.float64),
                          atol=1e-7)

    # (3) Held-out: arbitrary external directions match a manual Rayleigh quotient.
    Wext = torch.randn(zdim, zdim, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(180))
    h2_ext = her.heritability_of_directions(Z, Wext)
    gvar = torch.einsum("ik,ij,jk->k", Wext, G, Wext)
    pvar = torch.einsum("ik,ij,jk->k", Wext, P_reg, Wext)
    assert torch.allclose(h2_ext, gvar / pvar, atol=1e-8)
    print(f"  held-out directions OK  | self-spectrum recovered, WᵀP_regW=I, "
          f"external Rayleigh matches")


# ---------------------------------------------------------------------------
# Gradient through the encoder (rank-B path vs dense reference)
# ---------------------------------------------------------------------------

def test_gradient_through_encoder() -> None:
    n, m, zdim, B = 60, 40, 4, 10
    _, X, _, bed, row_idx = make_fixture(n, m, seed=1000)
    Z0 = torch.randn(n, zdim, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(2000))

    her = RankBHeritabilitySpectrum(bed, row_idx, C=None, ridge=RIDGE,
                                    dtype=torch.float64)
    patch_tr_pkp2_spectrum(her, X, C=None)
    her.rebuild(Z0)
    idxs = torch.randperm(n, generator=torch.Generator().manual_seed(3000))[:B]

    enc = torch.nn.Linear(n, zdim, bias=False).to(dtype=torch.float64)
    one_hot = torch.zeros(B, n, dtype=torch.float64)
    one_hot[torch.arange(B), idxs] = 1.0

    # Path A: rank-B.
    Z_batch_A = enc(one_hot)
    loss_A = her.update_and_loss(Z_batch_A, idxs)
    grad_A = torch.autograd.grad(loss_A, enc.weight)[0]

    # Path B: dense gcov_spectrum on the spliced full Z, uniform objective.
    Z_full = Z0.detach().clone()
    Z_batch_B = enc(one_hot)
    Z_full = Z_full.clone()
    Z_full[idxs] = Z_batch_B
    _, _, spec_B = gcov_spectrum(X, C=None, ridge=RIDGE)(Z_full)
    loss_B = -spec_B.sum()
    grad_B = torch.autograd.grad(loss_B, enc.weight)[0]

    assert torch.allclose(loss_A, loss_B, atol=1e-8), (loss_A, loss_B)
    assert torch.allclose(grad_A, grad_B, atol=1e-7), (
        f"max abs grad diff: {(grad_A - grad_B).abs().max().item():.3e}"
    )
    print(f"  encoder-side gradient match  | "
          f"max abs grad diff = {(grad_A - grad_B).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
# Many-step drift
# ---------------------------------------------------------------------------

def test_many_step_drift_with_C() -> None:
    n, m, zdim, c, B, T = 70, 45, 5, 4, 14, 30
    _, X, _, bed, row_idx = make_fixture(n, m, seed=400)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(500))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(600))

    her = RankBHeritabilitySpectrum(bed, row_idx, C=C, ridge=RIDGE,
                                    dtype=torch.float64)
    patch_tr_pkp2_spectrum(her, X, C=C)
    her.rebuild(Z)
    spectrum_fn = gcov_spectrum(X, C=C, ridge=RIDGE)

    gen = torch.Generator().manual_seed(700)
    Z_state = Z.clone()
    for t in range(T):
        idxs = torch.randperm(n, generator=gen)[:B]
        Z_new = torch.randn(B, zdim, dtype=torch.float64, generator=gen)
        Z_state = Z_state.clone()
        Z_state[idxs] = Z_new
        loss_rb = her.update_and_loss(Z_new, idxs)
        _, _, spec_ref = spectrum_fn(Z_state)
        loss_ref = -spec_ref.sum()
        assert torch.allclose(loss_rb, loss_ref, atol=1e-7), (
            f"step {t}: rank-B = {loss_rb.item()}, dense = {loss_ref.item()}"
        )
    print(f"  {T} steps, with-C, drift bounded  | final loss = {loss_rb.item():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # gc() / gcov_spectrum build intercept columns via torch.ones without an
    # explicit dtype in some paths; pin fp64 for the run.
    torch.set_default_dtype(torch.float64)
    print("RankBHeritabilitySpectrum tests:")
    test_spectrum_matches_dense_no_C()
    test_spectrum_matches_dense_with_C()
    test_G_matches_gc_pairwise_no_C()
    test_G_matches_gc_pairwise_with_C()
    test_objective_uniform_topk_clamp()
    test_combined_spectrum_marginal()
    test_heldout_directions()
    test_gradient_through_encoder()
    test_many_step_drift_with_C()
    print("all tests passed.")
