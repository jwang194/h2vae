"""Tests for the per-SNP sumstats helper.

Verifies:

- ``compute_sumstats`` produces β̂ matching a closed-form OLS on a
  materialised design matrix ``[X_j | W]``.
- χ² is consistent with ``(β̂ / SE)²``.
- After 30 overlapping rank-B updates, sumstats from the streaming
  state match a fresh full rebuild on the same Z snapshot.
- The fast path performs **no extra cache walks** per minibatch — only
  the cache walks that ``rebuild`` would have done anyway, plus the
  one-time ``d_j`` walk on first call.
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
from h2vae.rank_b_sumstats import compute_sumstats, update_and_sumstats  # noqa: E402
from fixtures import random_genotypes, write_plink                # noqa: E402


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def standardise_population(G: torch.Tensor) -> torch.Tensor:
    mu = G.mean(dim=0, keepdim=True)
    sd = G.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    return (G - mu) / sd


def make_fixture(n: int, m: int, seed: int):
    G_int8 = random_genotypes(n, m, seed=seed, missing_rate=0.0)
    G = torch.from_numpy(G_int8.astype(np.float64))
    X = standardise_population(G).to(torch.float64)
    sample_ids = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    tmp = tempfile.mkdtemp(prefix="rbs_")
    prefix = str(pathlib.Path(tmp) / "geno")
    write_plink(prefix, G_int8, sample_ids)
    return X, BedFile(prefix), np.arange(n, dtype=np.int64)


def standardise_sample(Z: torch.Tensor) -> torch.Tensor:
    mu = Z.mean(dim=0, keepdim=True)
    sd = Z.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-8)
    return (Z - mu) / sd


def ols_beta_per_snp(X: torch.Tensor, W: torch.Tensor | None,
                     Y: torch.Tensor) -> torch.Tensor:
    """Closed-form per-SNP OLS β̂ regressing each column of Y on (X_j, W).

    Returns ``(m, zdim)`` β̂ for the X_j coefficient (covariate
    coefficients are projected out).  Y is assumed pre-standardised.
    """
    n, m = X.shape
    zdim = Y.shape[1]
    beta = torch.zeros(m, zdim, dtype=X.dtype)
    if W is None:
        # β = X_j^T y / X_j^T X_j
        denom = (X * X).sum(dim=0)
        beta = (X.T @ Y) / denom[:, None]
        return beta
    # V = I - W (W'W)^{-1} W'  (symmetric, idempotent).
    WtW_inv = torch.linalg.inv(W.T @ W)
    PY = Y - W @ (WtW_inv @ (W.T @ Y))             # (n, zdim) = V Y
    PX = X - W @ (WtW_inv @ (W.T @ X))             # (n, m)    = V X
    XtPY = X.T @ PY                                # (m, zdim) = X^T V Y
    XtPX_diag = (X * PX).sum(dim=0)                # (m,)      = X_j^T V X_j
    return XtPY / XtPX_diag[:, None]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_beta_matches_ols_no_C() -> None:
    n, m, zdim = 60, 40, 4
    X, bed, row_idx = make_fixture(n, m, seed=1)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2))

    her = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    her.rebuild(Z)

    out = compute_sumstats(her)
    # Closed-form β̂ on standardised Z (rank-B uses sample-std internally).
    Zs = standardise_sample(Z)
    beta_ols = ols_beta_per_snp(X, None, Zs)

    assert torch.allclose(out["beta"], beta_ols, atol=1e-10), (
        f"\n  rank-B β: {out['beta'][:2]}\n  OLS β:    {beta_ols[:2]}"
    )
    # χ² consistency
    chisq_check = (out["z"] ** 2)
    assert torch.allclose(out["chisq"], chisq_check, atol=1e-12)
    print(f"  no-C   β̂ exact match  | max|Δ| = "
          f"{(out['beta'] - beta_ols).abs().max().item():.2e}")


def test_beta_matches_ols_with_C() -> None:
    n, m, zdim, c = 80, 50, 3, 4
    X, bed, row_idx = make_fixture(n, m, seed=10)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(20))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(30))

    her = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)
    her.rebuild(Z)

    out = compute_sumstats(her)
    Zs = standardise_sample(Z)
    beta_ols = ols_beta_per_snp(X, C, Zs)

    assert torch.allclose(out["beta"], beta_ols, atol=1e-9), (
        f"\n  rank-B β: {out['beta'][:2]}\n  OLS β:    {beta_ols[:2]}"
    )
    print(f"  with-C β̂ exact match  | max|Δ| = "
          f"{(out['beta'] - beta_ols).abs().max().item():.2e}")


def test_chisq_matches_z_squared() -> None:
    n, m, zdim = 50, 30, 2
    _, bed, row_idx = make_fixture(n, m, seed=99)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(98))
    her = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    her.rebuild(Z)
    out = compute_sumstats(her)
    assert torch.allclose(out["chisq"], (out["beta"] / out["se"]) ** 2, atol=1e-12)


def test_sumstats_fast_recompute() -> None:
    """30 rank-B updates: sumstats from the streaming state match a fresh rebuild."""
    n, m, zdim, c, B, T = 70, 45, 3, 4, 14, 30
    X, bed, row_idx = make_fixture(n, m, seed=111)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(222))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(333))

    her = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)
    her.rebuild(Z)

    # Reference instance — fully rebuilt each step.
    her_ref = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)

    gen = torch.Generator().manual_seed(444)
    Z_state = Z.clone()
    for t in range(T):
        idxs = torch.randperm(n, generator=gen)[:B]
        Z_new = torch.randn(B, zdim, dtype=torch.float64, generator=gen)
        Z_state = Z_state.clone()
        Z_state[idxs] = Z_new
        her.update_and_loss(Z_new, idxs)
        her_ref.rebuild(Z_state)

        out = compute_sumstats(her)
        out_ref = compute_sumstats(her_ref)

        assert torch.allclose(out["beta"], out_ref["beta"], atol=1e-9), (
            f"step {t}: β̂ drift max|Δ|={(out['beta']-out_ref['beta']).abs().max().item():.2e}"
        )
        assert torch.allclose(out["chisq"], out_ref["chisq"], atol=1e-7)
    print(f"  {T} steps: rank-B sumstats match fresh rebuild")


def test_no_extra_cache_walks_per_step() -> None:
    """``update_and_loss`` + ``compute_sumstats`` does not call _decode_chunk_std.

    The only chunked cache walks expected are:
      - setup (in __init__),
      - the one-time d_j walk on first compute_sumstats call,
      - one walk per rebuild (epoch start).
    Per-minibatch should use only ``_decode_rows_std`` (one sample-row gather).
    """
    n, m, zdim, B = 60, 40, 3, 12
    _, bed, row_idx = make_fixture(n, m, seed=555)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(666))

    her = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    her.rebuild(Z)
    compute_sumstats(her)   # warm the d_j cache.

    # Hook the chunk decoder; counter should not advance during the step.
    chunk_calls = [0]
    rows_calls = [0]
    orig_chunk = her._decode_chunk_std
    orig_rows = her._decode_rows_std

    def hooked_chunk(j_lo, j_hi):
        chunk_calls[0] += 1
        return orig_chunk(j_lo, j_hi)

    def hooked_rows(cohort_idx):
        rows_calls[0] += 1
        return orig_rows(cohort_idx)

    her._decode_chunk_std = hooked_chunk
    her._decode_rows_std = hooked_rows

    gen = torch.Generator().manual_seed(777)
    idxs = torch.randperm(n, generator=gen)[:B]
    Z_new = torch.randn(B, zdim, dtype=torch.float64, generator=gen)
    her.update_and_loss(Z_new, idxs)
    compute_sumstats(her)

    assert chunk_calls[0] == 0, (
        f"unexpected chunk-walk during step: {chunk_calls[0]} calls"
    )
    assert rows_calls[0] == 1, (
        f"expected exactly one row gather, saw {rows_calls[0]}"
    )
    print(f"  per-step: chunk_calls={chunk_calls[0]}  rows_calls={rows_calls[0]}")


def test_grad_flows_through_overrides() -> None:
    """compute_sumstats with grad-carrying u/Z overrides backprops correctly.

    ``update_and_loss`` detaches ``u_raw`` at each step's end so the
    autograd graph doesn't grow across steps; a wrapping module that
    wants grad through sumstats passes the un-detached ``u_new`` and
    ``Z`` snapshot via the override arguments.
    """
    n, m, zdim = 50, 30, 2
    X, bed, row_idx = make_fixture(n, m, seed=888)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(889),
                    requires_grad=True)

    her = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    her.rebuild(Z)
    # Construct u directly so it carries grad to Z.
    u_grad = X.to(torch.float64).T @ Z

    out = compute_sumstats(her, u_override=u_grad, Z_override=Z)
    loss = out["chisq"].sum()
    loss.backward()
    assert Z.grad is not None
    assert torch.isfinite(Z.grad).all()
    print(f"  grad max |∂L/∂Z| = {Z.grad.abs().max().item():.4f}")


def test_update_and_sumstats_grad_flow() -> None:
    """update_and_sumstats per-step: grad flows from Z_batch through χ²."""
    n, m, zdim, c, B = 60, 40, 3, 4, 14
    _, bed, row_idx = make_fixture(n, m, seed=901)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(902))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(903))

    her = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)
    her.rebuild(Z)

    idxs = torch.arange(B)
    Z_batch = Z[idxs].clone().detach().requires_grad_(True)
    out = update_and_sumstats(her, Z_batch, idxs)
    loss = out["chisq"].sum()
    loss.backward()
    assert Z_batch.grad is not None
    assert torch.isfinite(Z_batch.grad).all()
    assert Z_batch.grad.abs().max() > 0, "grad is identically zero"
    print(f"  update_and_sumstats grad max|∂L/∂Z| = "
          f"{Z_batch.grad.abs().max().item():.4f}")


def test_update_and_sumstats_matches_rebuild() -> None:
    """30 update_and_sumstats steps match fresh-rebuild sumstats on same Z."""
    n, m, zdim, c, B, T = 70, 45, 3, 4, 14, 30
    _, bed, row_idx = make_fixture(n, m, seed=904)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(905))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(906))

    her = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)
    her.rebuild(Z)
    her_ref = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)

    gen = torch.Generator().manual_seed(907)
    Z_state = Z.clone()
    for t in range(T):
        idxs = torch.randperm(n, generator=gen)[:B]
        Z_new = torch.randn(B, zdim, dtype=torch.float64, generator=gen)
        Z_state = Z_state.clone()
        Z_state[idxs] = Z_new
        out = update_and_sumstats(her, Z_new, idxs)
        her_ref.rebuild(Z_state)
        out_ref = compute_sumstats(her_ref)
        assert torch.allclose(out["beta"], out_ref["beta"], atol=1e-9), (
            f"step {t} β̂ drift"
        )
        assert torch.allclose(out["chisq"], out_ref["chisq"], atol=1e-7)
    print(f"  {T} steps via update_and_sumstats == fresh-rebuild sumstats")


if __name__ == "__main__":
    test_beta_matches_ols_no_C()
    test_beta_matches_ols_with_C()
    test_chisq_matches_z_squared()
    test_sumstats_fast_recompute()
    test_no_extra_cache_walks_per_step()
    test_grad_flows_through_overrides()
    test_update_and_sumstats_grad_flow()
    test_update_and_sumstats_matches_rebuild()
    print("ALL OK")
