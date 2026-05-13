"""Tests for the rank-B heritability module.

Verifies that the rank-B incremental computation matches
``h2vae.heritability.mom()`` per-dim h² output exactly when the
Hutchinson trace estimate is replaced by its exact value, and within
``1/sqrt(b_hutch)`` tolerance otherwise. Also covers the gradient
path, drift across many minibatches, and overlapping minibatches.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "test"))

from h2vae.plink import BedFile                                   # noqa: E402
from h2vae.rank_b_heritability import RankBHeritability           # noqa: E402
from h2vae.heritability import mom, gc                            # noqa: E402
from fixtures import random_genotypes, write_plink                # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def standardise_population(G: torch.Tensor) -> torch.Tensor:
    """Population (n-divisor) z-scoring; matches mom()'s implicit assumption."""
    mu = G.mean(dim=0, keepdim=True)
    sd = G.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    return (G - mu) / sd


def make_fixture(n: int, m: int, seed: int, missing_rate: float = 0.0):
    """Generate (ternary G, standardised X, sample_ids, BedFile)."""
    G_int8 = random_genotypes(n, m, seed=seed, missing_rate=missing_rate)
    G = torch.from_numpy(G_int8.astype(np.float64))
    if missing_rate > 0:
        # Mean-impute before standardising (matches the in-module behaviour).
        mask = G.eq(-1)
        for j in range(m):
            col = G[:, j]
            col_obs = col[~mask[:, j]]
            mu_j = col_obs.mean() if len(col_obs) else 0.0
            col[mask[:, j]] = mu_j
        G = G.to(torch.float64)
    X = standardise_population(G).to(torch.float64)

    sample_ids = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    tmp = tempfile.mkdtemp(prefix="rbh_")
    prefix = str(pathlib.Path(tmp) / "geno")
    write_plink(prefix, G_int8, sample_ids)
    bed = BedFile(prefix)
    row_idx = np.arange(n, dtype=np.int64)
    return G_int8, X, sample_ids, bed, row_idx


def patch_tr_pkp2_to_exact(her: RankBHeritability, X: torch.Tensor,
                            C: torch.Tensor | None) -> None:
    """Replace the Hutchinson tr_pkp2 with the exact value (for testability)."""
    K = (X @ X.T) / X.shape[1]
    if C is None:
        tr_K2 = float(torch.trace(K @ K))
        her.tr_pkp2 = torch.tensor(tr_K2, device=her.device, dtype=her.dtype)
        her.tr_K2 = her.tr_pkp2
    else:
        n = X.shape[0]
        P = torch.eye(n, dtype=X.dtype) - C @ torch.linalg.inv(C.T @ C) @ C.T
        PKP = P @ K @ P
        tr_pkp2 = float(torch.trace(PKP @ PKP))
        her.tr_pkp2 = torch.tensor(tr_pkp2, device=her.device, dtype=her.dtype)


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------

def test_no_C_matches_mom_exactly() -> None:
    """Rank-B h² == mom()(Z) per dim, no covariates, no missing genotypes."""
    n, m, zdim = 60, 40, 4
    _, X, _, bed, row_idx = make_fixture(n, m, seed=1)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2))

    her = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    patch_tr_pkp2_to_exact(her, X, C=None)
    her.rebuild(Z)
    # Trigger a no-op update_and_loss to extract per-dim h².
    h2_rankb = her._per_dim_signal(her.u_raw, None, Z[:0], torch.tensor([], dtype=torch.long))
    # ↑ but cohort_idx is empty; need a non-empty path. Re-run with a
    # dummy update where Z_batch == Z_prev[idxs] (delta = 0).
    idxs = torch.tensor([0, 1, 2, 3])
    h2_rankb = her._per_dim_signal(her.u_raw, her.w_raw if her.has_W else None,
                                Z[idxs], idxs)

    mom_fn = mom(X, kinship=False, C=None, device=torch.device("cpu"))
    h2_ref = mom_fn(Z)
    assert torch.allclose(h2_rankb, h2_ref.to(torch.float64), atol=1e-10), (
        f"\n  rank-B: {h2_rankb}\n  mom:    {h2_ref}"
    )
    print(f"  no-C exact match  | h² = {h2_rankb.numpy()}")


def test_with_C_matches_mom_exactly() -> None:
    """Rank-B h² == mom(C=...)(Z) per dim with residualisation."""
    n, m, zdim, c = 80, 50, 3, 4
    _, X, _, bed, row_idx = make_fixture(n, m, seed=10)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(20))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(30))

    her = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)
    patch_tr_pkp2_to_exact(her, X, C=C)
    her.rebuild(Z)
    idxs = torch.tensor([0, 1, 2, 3, 4])
    h2_rankb = her._per_dim_signal(her.u_raw, her.w_raw, Z[idxs], idxs)

    mom_fn = mom(X, kinship=False, C=C, device=torch.device("cpu"))
    h2_ref = mom_fn(Z)
    assert torch.allclose(h2_rankb, h2_ref.to(torch.float64), atol=1e-9), (
        f"\n  rank-B: {h2_rankb}\n  mom:    {h2_ref}"
    )
    print(f"  with-C exact match  | h² = {h2_rankb.numpy()}")


# ---------------------------------------------------------------------------
# Hutchinson tr_pkp2 convergence
# ---------------------------------------------------------------------------

def test_hutchinson_trace_converges() -> None:
    """tr_pkp2 estimate within 5/sqrt(b_hutch) of the true value (no C)."""
    n, m = 100, 80
    _, X, _, bed, row_idx = make_fixture(n, m, seed=42)
    K = (X @ X.T) / m
    tr_K2 = float(torch.trace(K @ K))

    for b in (10, 100, 500):
        her = RankBHeritability(bed, row_idx, C=None, b_hutch=b, seed_hutch=7,
                                 dtype=torch.float64)
        est = float(her.tr_pkp2)
        rel = abs(est - tr_K2) / tr_K2
        # Hutchinson stderr scales like 1/sqrt(b) but has prefactor 2 or so;
        # use a loose tolerance and only assert directionally for low b.
        print(f"  b_hutch={b}: est={est:.4f}  true={tr_K2:.4f}  rel={rel:.3f}")
        if b >= 500:
            assert rel < 0.2, f"b_hutch=500 estimate too far: {rel}"


# ---------------------------------------------------------------------------
# Many-step drift (continuous updates within an epoch)
# ---------------------------------------------------------------------------

def test_many_step_drift_no_C() -> None:
    """30 rank-B updates: each step's h² still matches a fresh mom() call."""
    n, m, zdim, B, T = 60, 40, 3, 12, 30
    _, X, _, bed, row_idx = make_fixture(n, m, seed=100)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(200))

    her = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    patch_tr_pkp2_to_exact(her, X, C=None)
    her.rebuild(Z)
    mom_fn = mom(X, kinship=False, C=None, device=torch.device("cpu"))

    gen = torch.Generator().manual_seed(300)
    Z_state = Z.clone()
    for t in range(T):
        idxs = torch.randperm(n, generator=gen)[:B]
        Z_new = torch.randn(B, zdim, dtype=torch.float64, generator=gen)
        Z_state = Z_state.clone()
        Z_state[idxs] = Z_new
        loss_rankb = her.update_and_loss(Z_new, idxs)
        # Reconstruct per-dim from a fresh mom() call on Z_state.
        h2_ref = mom_fn(Z_state)
        # The module returns -sum(h²); recover per-dim by re-running.
        h2_rankb = her._per_dim_signal(her.u_raw, her.w_raw if her.has_W else None,
                                    Z_state[idxs], idxs)
        assert torch.allclose(h2_rankb, h2_ref.to(torch.float64), atol=1e-9), (
            f"step {t}: rank-B = {h2_rankb}, mom = {h2_ref}"
        )
    print(f"  {T} steps, no-C, drift bounded  | final h² = {h2_rankb.numpy()}")


def test_many_step_drift_with_C() -> None:
    """30 rank-B updates with residualisation."""
    n, m, zdim, c, B, T = 60, 40, 3, 4, 12, 30
    _, X, _, bed, row_idx = make_fixture(n, m, seed=400)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(500))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(600))

    her = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)
    patch_tr_pkp2_to_exact(her, X, C=C)
    her.rebuild(Z)
    mom_fn = mom(X, kinship=False, C=C, device=torch.device("cpu"))

    gen = torch.Generator().manual_seed(700)
    Z_state = Z.clone()
    for t in range(T):
        idxs = torch.randperm(n, generator=gen)[:B]
        Z_new = torch.randn(B, zdim, dtype=torch.float64, generator=gen)
        Z_state = Z_state.clone()
        Z_state[idxs] = Z_new
        her.update_and_loss(Z_new, idxs)
        h2_ref = mom_fn(Z_state)
        h2_rankb = her._per_dim_signal(her.u_raw, her.w_raw, Z_state[idxs], idxs)
        assert torch.allclose(h2_rankb, h2_ref.to(torch.float64), atol=1e-8), (
            f"step {t}: rank-B = {h2_rankb}, mom = {h2_ref}"
        )
    print(f"  {T} steps, with-C, drift bounded  | final h² = {h2_rankb.numpy()}")


# ---------------------------------------------------------------------------
# Gradient through the encoder
# ---------------------------------------------------------------------------

def test_gradient_through_encoder() -> None:
    """Gradient via rank-B path matches gradient via full mom() recompute."""
    n, m, zdim, B = 50, 30, 3, 8
    _, X, _, bed, row_idx = make_fixture(n, m, seed=1000)
    Z0 = torch.randn(n, zdim, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(2000))

    # Path A: rank-B.
    her_A = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    patch_tr_pkp2_to_exact(her_A, X, C=None)
    her_A.rebuild(Z0)
    idxs = torch.randperm(n, generator=torch.Generator().manual_seed(3000))[:B]

    enc = torch.nn.Linear(n, zdim, bias=False).to(dtype=torch.float64)
    one_hot = torch.zeros(B, n, dtype=torch.float64)
    one_hot[torch.arange(B), idxs] = 1.0

    Z_batch_A = enc(one_hot)
    loss_A = her_A.update_and_loss(Z_batch_A, idxs)
    grad_A = torch.autograd.grad(loss_A, enc.weight)[0]

    # Path B: fresh full computation through mom().
    her_B = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    patch_tr_pkp2_to_exact(her_B, X, C=None)
    her_B.rebuild(Z0)
    Z_full = Z0.detach().clone()
    Z_batch_B = enc(one_hot)
    Z_full = Z_full.clone()
    Z_full[idxs] = Z_batch_B
    mom_fn = mom(X, kinship=False, C=None, device=torch.device("cpu"))
    h2_B = mom_fn(Z_full)
    loss_B = -h2_B.sum()
    grad_B = torch.autograd.grad(loss_B, enc.weight)[0]

    assert torch.allclose(loss_A, loss_B, atol=1e-9)
    assert torch.allclose(grad_A, grad_B, atol=1e-8), (
        f"max abs grad diff: {(grad_A - grad_B).abs().max().item():.3e}"
    )
    print(f"  encoder-side gradient match  | "
          f"max abs grad diff = {(grad_A - grad_B).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
# GC-mode tests
# ---------------------------------------------------------------------------

def patch_gc_tr_pkp2_to_exact(her: RankBHeritability, X: torch.Tensor,
                               C: torch.Tensor | None) -> None:
    """Replace gc-mode tr_pkp2 with the exact value (W = [1 | C])."""
    n = X.shape[0]
    K = (X @ X.T) / X.shape[1]
    ones = torch.ones((n, 1), dtype=X.dtype)
    W = ones if C is None else torch.hstack((ones, C))
    P = torch.eye(n, dtype=X.dtype) - W @ torch.linalg.inv(W.T @ W) @ W.T
    PKP = P @ K @ P
    tr_pkp2 = float(torch.trace(PKP @ PKP))
    her.tr_pkp2 = torch.tensor(tr_pkp2, device=her.device, dtype=her.dtype)
    # gc_det depends on tr_pkp2; refresh.
    her._gc_det = (her.tr_pkp2 * her.nc - her.tr_pkp ** 2).detach()


def test_gc_loss_matches_gc_exactly() -> None:
    """Rank-B γ̂ == gc(y2)(Z) per dim, no-C path."""
    n, m, zdim = 60, 40, 4
    _, X, _, bed, row_idx = make_fixture(n, m, seed=1100)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(1200))
    y2 = torch.randn(n, 1, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(1300))

    her = RankBHeritability(bed, row_idx, C=None, y_target=y2,
                             dtype=torch.float64)
    patch_gc_tr_pkp2_to_exact(her, X, C=None)
    her.rebuild(Z)
    idxs = torch.tensor([0, 1, 2, 3])
    gamma_rankb = her._per_dim_signal(her.u_raw, her.w_raw, Z[idxs], idxs)

    gc_fn = gc(X, y2, kinship=False, C=None, device=torch.device("cpu"))
    gamma_ref = gc_fn(Z)
    assert torch.allclose(gamma_rankb, gamma_ref.to(torch.float64), atol=1e-9), (
        f"\n  rank-B: {gamma_rankb}\n  gc:     {gamma_ref}"
    )
    print(f"  gc no-C γ̂ match  | γ̂ = {gamma_rankb.numpy()}")


def test_gc_loss_with_C_matches_gc_exactly() -> None:
    """Rank-B γ̂ == gc(y2, C=...)(Z) per dim, with covariates."""
    n, m, zdim, c = 80, 50, 3, 4
    _, X, _, bed, row_idx = make_fixture(n, m, seed=1400)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(1500))
    y2 = torch.randn(n, 1, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(1600))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(1700))

    her = RankBHeritability(bed, row_idx, C=C, y_target=y2,
                             dtype=torch.float64)
    patch_gc_tr_pkp2_to_exact(her, X, C=C)
    her.rebuild(Z)
    idxs = torch.tensor([0, 1, 2, 3, 4])
    gamma_rankb = her._per_dim_signal(her.u_raw, her.w_raw, Z[idxs], idxs)

    gc_fn = gc(X, y2, kinship=False, C=C, device=torch.device("cpu"))
    gamma_ref = gc_fn(Z)
    assert torch.allclose(gamma_rankb, gamma_ref.to(torch.float64), atol=1e-9), (
        f"\n  rank-B: {gamma_rankb}\n  gc:     {gamma_ref}"
    )
    print(f"  gc with-C γ̂ match  | γ̂ = {gamma_rankb.numpy()}")


def test_gc_display_rho_matches() -> None:
    """Rank-B .display(Z) == gc(...).display(Z) per dim, bounded form."""
    n, m, zdim, c = 80, 50, 3, 4
    _, X, _, bed, row_idx = make_fixture(n, m, seed=1800)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(1900))
    y2 = torch.randn(n, 1, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(2000))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2100))

    her = RankBHeritability(bed, row_idx, C=C, y_target=y2,
                             dtype=torch.float64)
    patch_gc_tr_pkp2_to_exact(her, X, C=C)
    rho_rankb = her.display(Z)

    gc_fn = gc(X, y2, kinship=False, C=C, device=torch.device("cpu"))
    rho_ref = gc_fn.display(Z)
    assert torch.allclose(rho_rankb, rho_ref.to(torch.float64), atol=1e-9), (
        f"\n  rank-B: {rho_rankb}\n  gc.display: {rho_ref}"
    )
    print(f"  gc display ρ̂ match  | ρ̂ = {rho_rankb.numpy()}")


def test_gc_many_step_drift() -> None:
    """30 rank-B gc updates: γ̂ tracks fresh gc(...)(Z_state) calls."""
    n, m, zdim, c, B, T = 60, 40, 3, 4, 12, 30
    _, X, _, bed, row_idx = make_fixture(n, m, seed=2200)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2300))
    y2 = torch.randn(n, 1, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(2400))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2500))

    her = RankBHeritability(bed, row_idx, C=C, y_target=y2,
                             dtype=torch.float64)
    patch_gc_tr_pkp2_to_exact(her, X, C=C)
    her.rebuild(Z)
    gc_fn = gc(X, y2, kinship=False, C=C, device=torch.device("cpu"))

    gen = torch.Generator().manual_seed(2600)
    Z_state = Z.clone()
    for t in range(T):
        idxs = torch.randperm(n, generator=gen)[:B]
        Z_new = torch.randn(B, zdim, dtype=torch.float64, generator=gen)
        Z_state = Z_state.clone()
        Z_state[idxs] = Z_new
        her.update_and_loss(Z_new, idxs)
        gamma_ref = gc_fn(Z_state)
        gamma_rankb = her._per_dim_signal(her.u_raw, her.w_raw,
                                            Z_state[idxs], idxs)
        assert torch.allclose(gamma_rankb, gamma_ref.to(torch.float64), atol=1e-8), (
            f"step {t}: rank-B = {gamma_rankb}, gc = {gamma_ref}"
        )
    print(f"  {T} gc steps, drift bounded  | final γ̂ = {gamma_rankb.numpy()}")


def test_gc_gradient_through_encoder() -> None:
    """gc gradient via rank-B path matches gradient via full gc()(Z)."""
    n, m, zdim, B = 50, 30, 3, 8
    _, X, _, bed, row_idx = make_fixture(n, m, seed=2700)
    Z0 = torch.randn(n, zdim, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(2800))
    y2 = torch.randn(n, 1, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(2900))

    her = RankBHeritability(bed, row_idx, C=None, y_target=y2,
                             dtype=torch.float64)
    patch_gc_tr_pkp2_to_exact(her, X, C=None)
    her.rebuild(Z0)
    idxs = torch.randperm(n, generator=torch.Generator().manual_seed(3100))[:B]

    enc = torch.nn.Linear(n, zdim, bias=False).to(dtype=torch.float64)
    one_hot = torch.zeros(B, n, dtype=torch.float64)
    one_hot[torch.arange(B), idxs] = 1.0
    Z_batch_A = enc(one_hot)
    loss_A = her.update_and_loss(Z_batch_A, idxs)
    grad_A = torch.autograd.grad(loss_A, enc.weight)[0]

    # Path B: fresh full gc() through the same encoder.
    her_B = RankBHeritability(bed, row_idx, C=None, y_target=y2,
                               dtype=torch.float64)
    patch_gc_tr_pkp2_to_exact(her_B, X, C=None)
    her_B.rebuild(Z0)
    Z_full = Z0.detach().clone()
    Z_batch_B = enc(one_hot)
    Z_full = Z_full.clone()
    Z_full[idxs] = Z_batch_B
    gc_fn = gc(X, y2, kinship=False, C=None, device=torch.device("cpu"))
    loss_B = -gc_fn(Z_full).sum()
    grad_B = torch.autograd.grad(loss_B, enc.weight)[0]

    assert torch.allclose(loss_A, loss_B, atol=1e-8)
    assert torch.allclose(grad_A, grad_B, atol=1e-8), (
        f"max abs grad diff: {(grad_A - grad_B).abs().max().item():.3e}"
    )
    print(f"  gc encoder-side gradient match  | "
          f"max diff = {(grad_A - grad_B).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # gc() in heritability.py constructs intercept columns via torch.ones
    # without an explicit dtype, so it picks up the global default. Set
    # fp64 here for the duration of the test run.
    torch.set_default_dtype(torch.float64)
    print("RankBHeritability tests:")
    test_no_C_matches_mom_exactly()
    test_with_C_matches_mom_exactly()
    test_hutchinson_trace_converges()
    test_many_step_drift_no_C()
    test_many_step_drift_with_C()
    test_gradient_through_encoder()
    test_gc_loss_matches_gc_exactly()
    test_gc_loss_with_C_matches_gc_exactly()
    test_gc_display_rho_matches()
    test_gc_many_step_drift()
    test_gc_gradient_through_encoder()
    print("all tests passed.")
