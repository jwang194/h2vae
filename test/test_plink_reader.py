"""Round-trip tests for the PLINK BedFile reader."""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "test"))

from h2vae.plink import BedFile, is_plink_prefix, read_fam_ids  # noqa: E402
from fixtures import random_genotypes, write_plink                # noqa: E402


def _make_fixture(n: int, m: int, seed: int = 0,
                  missing_rate: float = 0.0) -> tuple[str, np.ndarray, np.ndarray]:
    """Generate a tempdir-backed PLINK trio. Returns (prefix, X, sample_ids)."""
    X = random_genotypes(n, m, seed=seed, missing_rate=missing_rate)
    sample_ids = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    tmp = tempfile.mkdtemp(prefix="bedtest_")
    prefix = str(pathlib.Path(tmp) / "fixture")
    write_plink(prefix, X, sample_ids)
    return prefix, X, sample_ids


def test_round_trip_full_decode() -> None:
    """decode_variants on the full range matches the original X exactly."""
    prefix, X, sample_ids = _make_fixture(n=23, m=17, seed=1, missing_rate=0.0)
    bed = BedFile(prefix)
    assert bed.n_total == 23
    assert bed.m == 17
    assert np.array_equal(bed.sample_ids, sample_ids)
    Y = bed.decode_variants(0, bed.m)
    assert Y.shape == (23, 17)
    assert np.array_equal(Y, X), (Y, X)
    print(f"  full decode round-trip ok  (n=23, m=17)")


def test_round_trip_with_missing() -> None:
    """Missing genotypes survive the round-trip as -1."""
    prefix, X, _ = _make_fixture(n=64, m=8, seed=2, missing_rate=0.15)
    bed = BedFile(prefix)
    Y = bed.decode_variants(0, bed.m)
    assert np.array_equal(Y, X)
    assert (Y == -1).any(), "expected some missing genotypes"
    print(f"  missing-genotype round-trip ok  "
          f"({(Y == -1).sum()} missing of {Y.size})")


def test_decode_variants_chunk() -> None:
    """Decoding a sub-range matches the full matrix sliced."""
    prefix, X, _ = _make_fixture(n=40, m=20, seed=3)
    bed = BedFile(prefix)
    Y = bed.decode_variants(5, 15)
    assert Y.shape == (40, 10)
    assert np.array_equal(Y, X[:, 5:15])
    print(f"  variant-chunk decode ok  (j=[5, 15))")


def test_decode_variants_row_subset() -> None:
    """decode_variants with row_idx returns the right rows."""
    prefix, X, _ = _make_fixture(n=32, m=12, seed=4)
    bed = BedFile(prefix)
    rows = np.array([0, 5, 17, 31])
    Y = bed.decode_variants(0, bed.m, row_idx=rows)
    assert Y.shape == (4, 12)
    assert np.array_equal(Y, X[rows])
    print(f"  decode_variants row-subset ok")


def test_decode_rows_full_range() -> None:
    """decode_rows over the whole genome matches full decode + row select."""
    prefix, X, _ = _make_fixture(n=50, m=30, seed=5, missing_rate=0.05)
    bed = BedFile(prefix)
    rows = np.array([3, 17, 42, 49, 0])
    Y = bed.decode_rows(rows)
    assert Y.shape == (5, 30)
    assert np.array_equal(Y, X[rows])
    print(f"  decode_rows full-range ok")


def test_decode_rows_variant_range() -> None:
    """decode_rows on a variant sub-range matches the slice."""
    prefix, X, _ = _make_fixture(n=80, m=40, seed=6)
    bed = BedFile(prefix)
    rows = np.array([1, 2, 3, 79])
    Y = bed.decode_rows(rows, j_lo=10, j_hi=25)
    assert Y.shape == (4, 15)
    assert np.array_equal(Y, X[rows, 10:25])
    print(f"  decode_rows variant-range ok")


def test_is_plink_prefix_and_read_fam_ids() -> None:
    prefix, _, sample_ids = _make_fixture(n=10, m=5, seed=7)
    assert is_plink_prefix(prefix)
    assert not is_plink_prefix(prefix + "_does_not_exist")
    ids = read_fam_ids(prefix)
    assert np.array_equal(ids, sample_ids)
    print(f"  is_plink_prefix / read_fam_ids ok")


def test_n_total_not_multiple_of_four() -> None:
    """Edge case: n_samples not divisible by 4 (BED pads to byte boundary)."""
    for n in (1, 2, 3, 5, 7, 13, 31):
        prefix, X, _ = _make_fixture(n=n, m=4, seed=100 + n)
        bed = BedFile(prefix)
        Y = bed.decode_variants(0, bed.m)
        assert Y.shape == (n, 4), f"n={n}: got {Y.shape}"
        assert np.array_equal(Y, X), f"n={n} round-trip failed"
    print(f"  padded BED round-trips ok (n in {{1,2,3,5,7,13,31}})")


if __name__ == "__main__":
    print("BedFile tests:")
    test_round_trip_full_decode()
    test_round_trip_with_missing()
    test_decode_variants_chunk()
    test_decode_variants_row_subset()
    test_decode_rows_full_range()
    test_decode_rows_variant_range()
    test_is_plink_prefix_and_read_fam_ids()
    test_n_total_not_multiple_of_four()
    print("all tests passed.")
