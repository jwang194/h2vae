"""CUDA decode-and-standardise kernel correctness test.

Compares the output of the GPU kernel against the existing
``RankBHeritability._decode_chunk_std`` implementation (which already
returns standardised fp32 chunks on the configured device).  Verifies
bit-for-bit numerical agreement to fp32 precision on a small synthetic
fixture.
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

from h2vae.plink import BedFile                           # noqa: E402
from h2vae.cohort_cache import CohortCache                # noqa: E402
from h2vae.rank_b_heritability import RankBHeritability   # noqa: E402
from h2vae._decode_cuda import decode_and_standardise     # noqa: E402
from fixtures import random_genotypes, write_plink         # noqa: E402


def _make_fixture(n: int, m: int, seed: int = 0, missing_rate: float = 0.1):
    G = random_genotypes(n, m, seed=seed, missing_rate=missing_rate)
    sample_ids = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    tmp = tempfile.mkdtemp(prefix="cudadecode_")
    prefix = str(pathlib.Path(tmp) / "g")
    write_plink(prefix, G, sample_ids)
    return G, BedFile(prefix)


def test_decode_kernel_matches_python() -> None:
    """GPU kernel output ≡ RankBHeritability._decode_chunk_std (fp32)."""
    if not torch.cuda.is_available():
        print("  skipped (no CUDA)")
        return

    n, m = 256, 96
    G, bed = _make_fixture(n=n, m=m, seed=42, missing_rate=0.1)
    row_idx = np.arange(n)
    device = torch.device("cuda:0")

    # Reference: route through the existing CPU + GPU code path.
    her = RankBHeritability(bed, row_idx, C=None, device=device,
                             chunk_variants=16, b_hutch=5)
    cache = her.cache
    chunk_size = 16
    for j_lo in range(0, m, chunk_size):
        j_hi = min(j_lo + chunk_size, m)

        # Reference standardised fp32 chunk (current CPU-decode path).
        ref = her._decode_chunk_std(j_lo, j_hi)         # (n, chunk_var) fp32 on CUDA

        # New path: take the bit-packed slice, transfer to GPU, run kernel.
        byte_lo = j_lo // 4
        byte_hi = (j_hi + 3) // 4
        chunk_var = j_hi - j_lo
        packed = torch.from_numpy(
            np.ascontiguousarray(cache._cache[:, byte_lo:byte_hi])
        ).to(device, non_blocking=True)
        mean_chunk = her.var_mean[j_lo:j_hi].contiguous().to(torch.float32)
        sd_chunk = her.var_sd[j_lo:j_hi].contiguous().to(torch.float32)

        got = decode_and_standardise(packed, mean_chunk, sd_chunk,
                                      chunk_var=chunk_var)

        diff = (got.float() - ref.float()).abs().max().item()
        assert diff < 1e-5, (
            f"chunk j=[{j_lo},{j_hi}) — max abs diff {diff:.3e}"
        )

    print(f"  CUDA decode ≡ CPU decode  (n={n}, m={m}, all chunks)")


def test_partial_trailing_chunk() -> None:
    """m not divisible by 4 — last chunk has m % 4 leftover variants."""
    if not torch.cuda.is_available():
        print("  skipped (no CUDA)")
        return
    n, m = 128, 67          # 67 % 4 == 3
    G, bed = _make_fixture(n=n, m=m, seed=7)
    row_idx = np.arange(n)
    device = torch.device("cuda:0")
    her = RankBHeritability(bed, row_idx, C=None, device=device,
                             chunk_variants=16, b_hutch=3)
    j_lo, j_hi = 64, 67     # last partial chunk
    chunk_var = j_hi - j_lo
    ref = her._decode_chunk_std(j_lo, j_hi)
    byte_lo = j_lo // 4
    byte_hi = (j_hi + 3) // 4
    packed = torch.from_numpy(
        np.ascontiguousarray(her.cache._cache[:, byte_lo:byte_hi])
    ).to(device)
    mean_chunk = her.var_mean[j_lo:j_hi].contiguous().to(torch.float32)
    sd_chunk = her.var_sd[j_lo:j_hi].contiguous().to(torch.float32)
    got = decode_and_standardise(packed, mean_chunk, sd_chunk,
                                  chunk_var=chunk_var)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    diff = (got.float() - ref.float()).abs().max().item()
    assert diff < 1e-5, f"partial chunk diff {diff:.3e}"
    print(f"  partial-chunk match  (chunk_var={chunk_var})")


if __name__ == "__main__":
    print("CUDA decode-and-standardise kernel tests:")
    test_decode_kernel_matches_python()
    test_partial_trailing_chunk()
    print("all tests passed.")
