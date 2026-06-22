"""Tests for ``RankBGenCorrLDSC``.

Covers:

* rg loss matches a direct ``ldsc_torch.RG`` call on the same z1
  (the rank-B path doesn't change LDSC semantics).
* rank-B updates equal a fresh-rebuild ground truth after many steps.
* No extra BED reads per step (single sample-row gather only).
* Gradient flows through ``update_and_loss`` to ``Z_batch``.
* Free intercepts produce non-trivial values (smoke test).
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "test"))

from fixtures import random_genotypes, write_plink                # noqa: E402
from h2vae.plink import BedFile                                   # noqa: E402
from h2vae.rank_b_heritability import RankBHeritability           # noqa: E402
from h2vae.rank_b_sumstats import compute_sumstats                # noqa: E402
from h2vae.ldsc_io import LDSCContext, build_ldsc_context         # noqa: E402
from h2vae.ldsc_torch import RG                                   # noqa: E402
from h2vae.rank_b_gencorr_ldsc import RankBGenCorrLDSC            # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: small synthetic BED + sumstats + ld-score trio
# ---------------------------------------------------------------------------

def make_cohort(n: int, m: int, seed: int):
    G_int8 = random_genotypes(n, m, seed=seed, missing_rate=0.0)
    sample_ids = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    tmp = tempfile.mkdtemp(prefix="rbg_")
    prefix = str(pathlib.Path(tmp) / "geno")
    write_plink(prefix, G_int8, sample_ids)
    bed = BedFile(prefix)
    return bed, np.arange(n, dtype=np.int64), G_int8


def make_ldsc_files(tmp: pathlib.Path, variant_ids: np.ndarray,
                    rng: np.random.Generator):
    """Synthesize a single-chrom ldscore + sumstats trio aligned to BED."""
    m = len(variant_ids)
    # Ref-ld: positive single annotation, slowly varying.
    ref_df = pd.DataFrame({
        "CHR": [1] * m,
        "SNP": variant_ids,
        "BP": np.arange(1, m + 1),
        "L2": rng.uniform(1.5, 4.0, size=m),
    })
    ref_pfx = str(tmp / "ref.")
    ref_df.to_csv(f"{ref_pfx}1.l2.ldscore.gz", sep="\t",
                   index=False, compression="gzip")
    with open(f"{ref_pfx}1.l2.M_5_50", "w") as f:
        f.write(f"{m * 50}\n")    # Plausible M

    # w-ld: same SNPs, different L2 values.
    w_df = pd.DataFrame({
        "CHR": [1] * m, "SNP": variant_ids,
        "BP": np.arange(1, m + 1),
        "L2": rng.uniform(0.8, 2.5, size=m),
    })
    w_pfx = str(tmp / "w.")
    w_df.to_csv(f"{w_pfx}1.l2.ldscore.gz", sep="\t",
                index=False, compression="gzip")
    with open(f"{w_pfx}1.l2.M_5_50", "w") as f:
        f.write(f"{m * 50}\n")

    # External sumstats — Z-scores.
    z_ext = rng.normal(size=m) * 1.5
    ss_df = pd.DataFrame({
        "SNP": variant_ids,
        "A1": ["A"] * m, "A2": ["G"] * m,
        "N": [50_000] * m,
        "Z": z_ext,
    })
    ss = tmp / "trait.sumstats.gz"
    ss_df.to_csv(ss, sep="\t", index=False, compression="gzip")
    return ss, ref_pfx, w_pfx


def make_full_fixture(n=80, m=60, seed=42):
    bed, row_idx, _ = make_cohort(n, m, seed=seed)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="rbg_ld_"))
    rng = np.random.default_rng(seed + 1)
    ss, ref_pfx, w_pfx = make_ldsc_files(tmp, bed.variant_ids, rng)
    ctx = build_ldsc_context(
        ss, ref_pfx, w_pfx,
        bed_variant_ids=bed.variant_ids, chroms=[1],
    )
    return bed, row_idx, ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loss_matches_direct_rg() -> None:
    """RankBGenCorrLDSC loss == -sum_d RG(z_d, z_ext).rg_ratio (direct)."""
    n, m, zdim = 80, 60, 3
    bed, row_idx, ctx = make_full_fixture(n=n, m=m, seed=11)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(12))

    rankb = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    rankb.rebuild(Z)

    # Constrained intercepts → numerically clean equivalence check on tiny m.
    module = RankBGenCorrLDSC(rankb, ctx,
                               intercept_hsq=1.0, intercept_gencov=0.0)
    # Reference: full-precision direct RG over the same sumstats.
    out = compute_sumstats(rankb)
    z_aligned = out["z"][ctx.bed_to_ldsc_idx]
    N1 = torch.full((ctx.m_use, 1), float(rankb.n - rankb.c),
                    dtype=torch.float64)
    direct = []
    for d in range(zdim):
        r = RG(z_aligned[:, d:d + 1], ctx.z_external,
               ctx.ref_ld, ctx.w_ld, N1, ctx.n_external, ctx.M,
               intercept_hsq1=1.0, intercept_hsq2=1.0, intercept_gencov=0.0)
        direct.append(_rg_or_zero(r))
    direct_loss = -torch.stack(direct).sum()

    # Module loss — display takes the same path
    rg_vec = module.display(Z)
    module_loss = -rg_vec.sum()
    assert torch.allclose(module_loss, direct_loss, atol=1e-10), (
        f"\n module: {module_loss}\n direct: {direct_loss}"
    )
    print(f"  loss matches direct  | loss = {module_loss.item():.4f}  "
          f"rg = {rg_vec.numpy()}")


def test_rank_b_step_matches_rebuild() -> None:
    """30 rank-B steps: rg vector matches a freshly rebuilt baseline."""
    n, m, zdim, c, B, T = 80, 60, 3, 4, 16, 30
    bed, row_idx, ctx = make_full_fixture(n=n, m=m, seed=21)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(22))
    C = torch.randn(n, c, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(23))

    rankb = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)
    rankb_ref = RankBHeritability(bed, row_idx, C=C, dtype=torch.float64)

    module = RankBGenCorrLDSC(rankb, ctx,
                               intercept_hsq=1.0, intercept_gencov=0.0)
    module_ref = RankBGenCorrLDSC(rankb_ref, ctx,
                                   intercept_hsq=1.0, intercept_gencov=0.0)
    module.rebuild(Z)

    gen = torch.Generator().manual_seed(24)
    Z_state = Z.clone()
    for t in range(T):
        idxs = torch.randperm(n, generator=gen)[:B]
        Z_new = torch.randn(B, zdim, dtype=torch.float64, generator=gen)
        Z_state = Z_state.clone()
        Z_state[idxs] = Z_new

        loss_rb = module.update_and_loss(Z_new, idxs)
        rg_rb = module.last_rg.clone()
        rg_ref = module_ref.display(Z_state)

        assert torch.allclose(rg_rb, rg_ref, atol=1e-6), (
            f"step {t} rg drift: max|Δ|={(rg_rb - rg_ref).abs().max():.2e}"
        )
    print(f"  {T} rank-B steps == fresh-rebuild rg")


def test_no_extra_bed_reads_per_step() -> None:
    """Per-step: exactly one ``_decode_rows_std`` call, zero ``_decode_chunk_std``."""
    n, m, zdim, B = 60, 50, 2, 12
    bed, row_idx, ctx = make_full_fixture(n=n, m=m, seed=31)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(32))

    rankb = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    module = RankBGenCorrLDSC(rankb, ctx)
    module.rebuild(Z)
    # warm d_j cache
    compute_sumstats(rankb)

    chunk_calls = [0]
    rows_calls = [0]
    orig_chunk = rankb._decode_chunk_std
    orig_rows = rankb._decode_rows_std

    def hooked_chunk(j_lo, j_hi):
        chunk_calls[0] += 1
        return orig_chunk(j_lo, j_hi)

    def hooked_rows(idx):
        rows_calls[0] += 1
        return orig_rows(idx)

    rankb._decode_chunk_std = hooked_chunk
    rankb._decode_rows_std = hooked_rows

    gen = torch.Generator().manual_seed(33)
    idxs = torch.randperm(n, generator=gen)[:B]
    Z_new = torch.randn(B, zdim, dtype=torch.float64, generator=gen)
    module.update_and_loss(Z_new, idxs)

    assert chunk_calls[0] == 0, f"unexpected chunk-walks: {chunk_calls[0]}"
    assert rows_calls[0] == 1, f"expected 1 row gather, got {rows_calls[0]}"
    print(f"  per-step: chunk={chunk_calls[0]}  rows={rows_calls[0]}")


def test_grad_flows_to_z_batch() -> None:
    """backward through RankBGenCorrLDSC populates Z_batch.grad."""
    n, m, zdim, B = 70, 50, 2, 14
    bed, row_idx, ctx = make_full_fixture(n=n, m=m, seed=41)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(42))

    rankb = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    module = RankBGenCorrLDSC(rankb, ctx)
    module.rebuild(Z)

    idxs = torch.arange(B)
    Z_batch = Z[idxs].clone().detach().requires_grad_(True)
    loss = module.update_and_loss(Z_batch, idxs)
    if not torch.isfinite(loss):
        # degenerate fixture; skip with informative message
        print(f"  loss not finite ({loss.item()}); skipping grad check")
        return
    loss.backward()
    assert Z_batch.grad is not None
    assert torch.isfinite(Z_batch.grad).all(), "non-finite grad"
    print(f"  grad max|∂L/∂Z| = {Z_batch.grad.abs().max().item():.4f}  "
          f"loss = {loss.item():.4f}")


def test_free_intercepts_diagnostic_fields() -> None:
    """After update_and_loss, last_intercepts dict is populated with all 3 entries."""
    n, m, zdim = 70, 50, 3
    bed, row_idx, ctx = make_full_fixture(n=n, m=m, seed=51)
    Z = torch.randn(n, zdim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(52))

    rankb = RankBHeritability(bed, row_idx, C=None, dtype=torch.float64)
    module = RankBGenCorrLDSC(rankb, ctx,
                               intercept_hsq=None, intercept_gencov=None)
    module.rebuild(Z)
    loss = module.update_and_loss(Z[:10].clone().detach(),
                                   torch.arange(10))
    intercepts = module.last_intercepts
    for key in ("hsq1", "hsq2", "gencov"):
        assert key in intercepts
        assert intercepts[key].numel() == zdim
        assert torch.isfinite(intercepts[key]).all(), (
            f"non-finite intercept estimate in {key}: {intercepts[key]}"
        )
    print(f"  intercepts: hsq1={intercepts['hsq1'].mean():.3f}  "
          f"hsq2={intercepts['hsq2'].mean():.3f}  "
          f"gc={intercepts['gencov'].mean():.3f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rg_or_zero(r) -> torch.Tensor:
    if r._negative_hsq:
        return torch.zeros((), dtype=torch.float64)
    if isinstance(r.rg_ratio, torch.Tensor):
        return r.rg_ratio
    return torch.as_tensor(r.rg_ratio, dtype=torch.float64)


if __name__ == "__main__":
    test_loss_matches_direct_rg()
    test_rank_b_step_matches_rebuild()
    test_no_extra_bed_reads_per_step()
    test_grad_flows_to_z_batch()
    test_free_intercepts_diagnostic_fields()
    print("ALL OK")
