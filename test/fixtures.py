"""Helpers to generate synthetic PLINK trios for tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Inverse of _BED_TABLE: genotype value → 2-bit code.
# 0 -> 0b11, 1 -> 0b10, 2 -> 0b00, missing(-1) -> 0b01
_GENO_TO_CODE = {0: 0b11, 1: 0b10, 2: 0b00, -1: 0b01}
_BED_MAGIC = bytes((0x6C, 0x1B, 0x01))


def write_plink(prefix: str | Path, X: np.ndarray, sample_ids: np.ndarray,
                variant_ids: np.ndarray | None = None) -> None:
    """Write a SNP-major BED/BIM/FAM trio.

    Args:
        prefix: output path prefix (no extension).
        X: ``(n, m)`` int8 array, values in ``{0, 1, 2, -1}``.
        sample_ids: ``(n,)`` integer IIDs.
        variant_ids: ``(m,)`` strings; defaults to ``rs0`` … ``rs{m-1}``.
    """
    n, m = X.shape
    if variant_ids is None:
        variant_ids = np.array([f"rs{j}" for j in range(m)], dtype=object)

    # NOTE: Path.with_suffix replaces only the LAST extension, so it would
    # turn "geno.even" → "geno.bed".  Concatenate strings instead.
    prefix_str = str(prefix)
    bed_path = Path(prefix_str + ".bed")
    bim_path = Path(prefix_str + ".bim")
    fam_path = Path(prefix_str + ".fam")

    # FAM: fid iid pat mat sex pheno
    with open(fam_path, "w") as f:
        for iid in sample_ids:
            f.write(f"{iid} {iid} 0 0 0 -9\n")

    # BIM: chrom varid genpos bp a1 a2
    with open(bim_path, "w") as f:
        for j, vid in enumerate(variant_ids):
            f.write(f"1 {vid} 0 {j+1} A G\n")

    # BED: magic + ceil(n/4) bytes per variant.
    bytes_per_var = (n + 3) // 4
    out = np.zeros((m, bytes_per_var), dtype=np.uint8)
    for j in range(m):
        col = X[:, j]
        for i in range(n):
            code = _GENO_TO_CODE[int(col[i])]
            out[j, i // 4] |= (code & 0b11) << (2 * (i % 4))
    with open(bed_path, "wb") as f:
        f.write(_BED_MAGIC)
        out.tofile(f)


def random_genotypes(n: int, m: int, seed: int = 0,
                     missing_rate: float = 0.0) -> np.ndarray:
    """Random ternary genotype matrix, optionally with missing entries."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    if missing_rate > 0:
        mask = rng.random((n, m)) < missing_rate
        X[mask] = -1
    return X
