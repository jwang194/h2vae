"""Round-trip tests for the bit-packed CohortCache."""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "test"))

from h2vae.cohort_cache import CohortCache               # noqa: E402
from h2vae.plink import BedFile                          # noqa: E402
from fixtures import random_genotypes, write_plink        # noqa: E402


def _build_cache_from_bed(bed: BedFile, row_idx: np.ndarray,
                           chunk: int = 16) -> CohortCache:
    """Helper: mirror RankBHeritability's intended build pattern."""
    cache = CohortCache(n=len(row_idx), m=bed.m, chunk_variants=chunk)
    j = 0
    while j < bed.m:
        j_hi = min(j + chunk, bed.m)
        X_int8 = bed.decode_variants(j, j_hi, row_idx=row_idx)
        cache.build_chunk(j, j_hi, X_int8)
        j = j_hi
    cache.finalise()
    return cache


def _make_fixture(n: int, m: int, seed: int, missing_rate: float = 0.0):
    G = random_genotypes(n, m, seed=seed, missing_rate=missing_rate)
    sample_ids = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    tmp = tempfile.mkdtemp(prefix="cache_")
    prefix = str(pathlib.Path(tmp) / "geno")
    write_plink(prefix, G, sample_ids)
    return G, BedFile(prefix)


def test_full_variant_chunk_round_trip() -> None:
    G, bed = _make_fixture(n=32, m=40, seed=1)
    row_idx = np.arange(32)
    cache = _build_cache_from_bed(bed, row_idx, chunk=16)
    Y = cache.decode_variant_chunk(0, 40)
    assert Y.shape == G.shape
    assert np.array_equal(Y, G), (Y - G)
    print(f"  full variant decode round-trip ok  (n=32, m=40)")


def test_partial_chunk_round_trip() -> None:
    """m not divisible by chunk — last chunk is partial."""
    G, bed = _make_fixture(n=24, m=37, seed=2)
    row_idx = np.arange(24)
    cache = _build_cache_from_bed(bed, row_idx, chunk=16)
    Y = cache.decode_variant_chunk(0, 37)
    assert Y.shape == G.shape
    assert np.array_equal(Y, G)
    print(f"  partial last chunk round-trip ok  (m=37 with chunk=16)")


def test_missing_genotypes_preserved() -> None:
    G, bed = _make_fixture(n=64, m=24, seed=3, missing_rate=0.2)
    row_idx = np.arange(64)
    cache = _build_cache_from_bed(bed, row_idx, chunk=8)
    Y = cache.decode_variant_chunk(0, 24)
    assert np.array_equal(Y, G)
    assert (Y == -1).any(), "expected missing values"
    print(f"  missing-genotype round-trip ok  "
          f"({(Y == -1).sum()} missing of {Y.size})")


def test_cohort_subset() -> None:
    """The cache is built over a subset of BED rows."""
    G, bed = _make_fixture(n=64, m=20, seed=4)
    row_idx = np.array([5, 10, 15, 20, 25, 30, 35, 40])
    cache = _build_cache_from_bed(bed, row_idx, chunk=8)
    Y = cache.decode_variant_chunk(0, 20)
    assert Y.shape == (8, 20)
    assert np.array_equal(Y, G[row_idx])
    print(f"  cohort-subset cache ok  (8 of 64 BED rows)")


def test_decode_rows_matches_full() -> None:
    G, bed = _make_fixture(n=48, m=32, seed=5)
    row_idx = np.arange(48)
    cache = _build_cache_from_bed(bed, row_idx, chunk=16)
    rows = np.array([0, 7, 23, 47, 17])
    Y = cache.decode_rows(rows)
    assert Y.shape == (5, 32)
    assert np.array_equal(Y, G[rows])
    print(f"  decode_rows full-genome ok")


def test_decode_rows_variant_range() -> None:
    G, bed = _make_fixture(n=64, m=40, seed=6)
    row_idx = np.arange(64)
    cache = _build_cache_from_bed(bed, row_idx, chunk=8)
    rows = np.array([1, 2, 3, 63])
    Y = cache.decode_rows(rows, j_lo=8, j_hi=24)
    assert Y.shape == (4, 16)
    assert np.array_equal(Y, G[rows, 8:24])
    print(f"  decode_rows variant-range ok")


def test_build_before_finalise_blocks_read() -> None:
    cache = CohortCache(n=4, m=8, chunk_variants=4)
    cache.build_chunk(0, 4, np.zeros((4, 4), dtype=np.int8))
    cache.build_chunk(4, 8, np.zeros((4, 4), dtype=np.int8))
    try:
        cache.decode_variant_chunk(0, 8)
    except RuntimeError as e:
        assert "finalise" in str(e)
        print(f"  read-before-finalise raises  ok")
    else:
        raise AssertionError("expected RuntimeError")


def test_chunk_must_be_4_aligned() -> None:
    try:
        CohortCache(n=4, m=8, chunk_variants=5)
    except ValueError as e:
        assert "4" in str(e)
        print(f"  non-4-aligned chunk_variants raises  ok")
    else:
        raise AssertionError("expected ValueError")


def test_fast_kernel_matches_slow_path() -> None:
    """C-kernel build path agrees with the numpy build path bit-for-bit."""
    from h2vae.cohort_cache import have_fast_kernel
    if not have_fast_kernel():
        print(f"  fast-kernel test skipped (no _bed_pack.so)")
        return

    G, bed = _make_fixture(n=64, m=40, seed=99, missing_rate=0.15)
    row_idx = np.array([3, 11, 17, 22, 29, 37, 50, 58, 63, 0])

    # --- Slow path ---
    slow = CohortCache(n=len(row_idx), m=bed.m, chunk_variants=16)
    chunk = 16
    j = 0
    while j < bed.m:
        j_hi = min(j + chunk, bed.m)
        X_int8 = bed.decode_variants(j, j_hi, row_idx=row_idx)
        slow.build_chunk(j, j_hi, X_int8)
        j = j_hi
    slow.finalise()

    # --- Fast path ---
    fast = CohortCache(n=len(row_idx), m=bed.m, chunk_variants=16)
    bytes_per_var_BED = bed.bytes_per_variant
    j = 0
    while j < bed.m:
        j_hi = min(j + chunk, bed.m)
        n_var = j_hi - j
        # mmap slice of the BED.
        bed_block = bed._mm[j:j_hi]               # (n_var, bytes_per_var_BED)
        sum_x = np.zeros(n_var, dtype=np.int64)
        sum_x2 = np.zeros(n_var, dtype=np.int64)
        n_obs = np.zeros(n_var, dtype=np.int64)
        if j_hi == bed.m and (j_hi - j) % 4 != 0:
            # For the partial last chunk, pad to a multiple of 4 with
            # missing variants — but the slow path also handles partial
            # natively. Here we just skip this test for the unaligned tail.
            break
        fast.build_chunk_from_bed(bed_block, row_idx, j, sum_x, sum_x2, n_obs)
        j = j_hi
    fast.finalise()

    # Decode both caches and compare.
    # If we stopped early due to partial-tail, decode up to the aligned point.
    j_final = j
    Y_slow = slow.decode_variant_chunk(0, j_final)
    Y_fast = fast.decode_variant_chunk(0, j_final)
    assert np.array_equal(Y_slow, Y_fast), (
        f"fast/slow mismatch at first diff: "
        f"{np.argmax((Y_slow != Y_fast).any(axis=0))}"
    )
    print(f"  fast C kernel ≡ slow numpy path  "
          f"(j_final={j_final}, n_cohort={len(row_idx)})")


if __name__ == "__main__":
    print("CohortCache tests:")
    test_full_variant_chunk_round_trip()
    test_partial_chunk_round_trip()
    test_missing_genotypes_preserved()
    test_cohort_subset()
    test_decode_rows_matches_full()
    test_decode_rows_variant_range()
    test_build_before_finalise_blocks_read()
    test_chunk_must_be_4_aligned()
    test_fast_kernel_matches_slow_path()
    print("all tests passed.")
