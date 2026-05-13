"""Bit-packed sample-major in-memory cohort × variant genotype cache.

Built once at ``RankBHeritability`` construction (piggybacking on the
variant-stats BED pass), this cache replaces all subsequent BED reads.
Per the analysis in ``notes/rank_b_heritability_perf.md`` (combined
options 5+6+8), it brings the rank-B path's per-fit BED traffic in
line with SCORE's Round 1: **one disk pass, ever**.

Layout
------
``_cache`` has shape ``(n_cohort, ceil(m / 4))`` ``uint8``. Each byte
packs 4 consecutive variants for one sample at 2 bits per genotype.
The encoding is:

============= =================
``0b00``      genotype 0 (homozygous reference)
``0b01``      genotype 1 (heterozygous)
``0b10``      genotype 2 (homozygous alternate)
``0b11``      missing (mean-imputed at decode time)
============= =================

This is **sample-major**: rows of the same sample are contiguous, so
``decode_rows(idxs)`` is a fancy-index row gather (B contiguous
``ceil(m/4)``-byte strips). Variant-chunked access slices the column
range ``[j_lo//4, ceil(j_hi/4))`` and is also sequential within each
row. Both access patterns therefore hit memory bandwidth limits, not
random-access limits.

Memory budget
-------------
At n=127k, m=613k: ``19.5 GB`` per cohort. Two cohorts (train+val):
``39 GB``. Comfortably fits on standard 256 GB nodes.
"""
from __future__ import annotations

import numpy as np

# Code → genotype lookup. Indexed by 2-bit code value.
_DECODE_TABLE = np.array([0, 1, 2, -1], dtype=np.int8)


class CohortCache:
    """Sample-major bit-packed genotype cache for one cohort.

    Build pattern (must be in order):

    .. code-block:: python

        cache = CohortCache(n=n_cohort, m=m_variants, chunk_variants=4096)
        for j_lo, j_hi, X_int8 in variant_chunks(...):
            ...  # update other accumulators (variant stats etc.)
            cache.build_chunk(j_lo, j_hi, X_int8)
        cache.finalise()

    After ``finalise()``, all reads happen via ``decode_variant_chunk``
    or ``decode_rows``.

    Args:
        n: cohort size.
        m: total variant count.
        chunk_variants: variant chunk size used during build. Must be a
            multiple of 4 so chunk byte boundaries stay aligned.
    """

    def __init__(self, n: int, m: int, chunk_variants: int = 4096):
        if chunk_variants % 4 != 0:
            raise ValueError(
                f"chunk_variants must be a multiple of 4, got {chunk_variants}"
            )
        self.n = int(n)
        self.m = int(m)
        self.chunk = int(chunk_variants)
        self.bytes_per_sample = (self.m + 3) // 4
        self._cache = np.zeros((self.n, self.bytes_per_sample), dtype=np.uint8)
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_chunk(self, j_lo: int, j_hi: int, X_int8: np.ndarray) -> None:
        """Pack one decoded chunk into the cache.

        Args:
            j_lo: start of the variant range (must be a multiple of 4,
                except for the trailing partial chunk).
            j_hi: end of the variant range (exclusive).
            X_int8: ``(n, j_hi - j_lo)`` int8 array with values in
                ``{-1, 0, 1, 2}``.
        """
        if self._built:
            raise RuntimeError("CohortCache already finalised")
        if j_lo % 4 != 0:
            raise ValueError(
                f"j_lo must be 4-aligned to keep chunk byte writes "
                f"non-overlapping (got {j_lo})"
            )
        n, c = X_int8.shape
        if n != self.n or c != j_hi - j_lo:
            raise ValueError(
                f"chunk shape mismatch: expected ({self.n}, {j_hi - j_lo}), "
                f"got {X_int8.shape}"
            )
        # Encode -1 (missing) as 0b11; 0, 1, 2 stay as themselves.
        codes = np.where(X_int8 == -1, 3, X_int8).astype(np.uint8)
        # Pad the last (possibly partial) chunk up to a multiple of 4
        # variants, with missing values for the padding.
        pad = (4 - (c % 4)) % 4
        if pad:
            codes = np.pad(codes, ((0, 0), (0, pad)), constant_values=3)
        codes = codes.reshape(self.n, -1, 4)
        packed = (codes[:, :, 0]
                  | (codes[:, :, 1] << 2)
                  | (codes[:, :, 2] << 4)
                  | (codes[:, :, 3] << 6)).astype(np.uint8)
        byte_lo = j_lo // 4
        self._cache[:, byte_lo:byte_lo + packed.shape[1]] = packed

    def finalise(self) -> None:
        """Mark the cache as fully populated and ready for reads."""
        self._built = True

    @property
    def nbytes(self) -> int:
        return int(self._cache.nbytes)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def decode_variant_chunk(self, j_lo: int, j_hi: int) -> np.ndarray:
        """Return the ``(n, j_hi - j_lo)`` int8 genotype block from cache."""
        self._check_built()
        if j_lo % 4 != 0:
            raise ValueError(f"j_lo must be 4-aligned, got {j_lo}")
        byte_lo = j_lo // 4
        byte_hi = (j_hi + 3) // 4
        packed = self._cache[:, byte_lo:byte_hi]                   # view
        decoded = self._unpack(packed)
        return decoded[:, :(j_hi - j_lo)]

    def decode_rows(self, row_idx: np.ndarray,
                    j_lo: int = 0, j_hi: int | None = None) -> np.ndarray:
        """Return ``(len(row_idx), j_hi - j_lo)`` int8 for the given rows.

        ``j_lo`` defaults to 0 and ``j_hi`` to ``self.m`` (whole genome).
        """
        self._check_built()
        if j_hi is None:
            j_hi = self.m
        if j_lo % 4 != 0:
            raise ValueError(f"j_lo must be 4-aligned, got {j_lo}")
        row_idx = np.asarray(row_idx, dtype=np.int64)
        if row_idx.ndim != 1:
            raise ValueError("row_idx must be 1-D")
        byte_lo = j_lo // 4
        byte_hi = (j_hi + 3) // 4
        packed = self._cache[row_idx, byte_lo:byte_hi]             # fancy idx (copy)
        decoded = self._unpack(packed)
        return decoded[:, :(j_hi - j_lo)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_built(self) -> None:
        if not self._built:
            raise RuntimeError("CohortCache not finalised — call .finalise()")

    @staticmethod
    def _unpack(packed: np.ndarray) -> np.ndarray:
        """Unpack a (rows, bytes) uint8 buffer to (rows, bytes*4) int8."""
        rows, nb = packed.shape
        codes = np.empty((rows, nb * 4), dtype=np.uint8)
        codes[:, 0::4] = packed & 0x3
        codes[:, 1::4] = (packed >> 2) & 0x3
        codes[:, 2::4] = (packed >> 4) & 0x3
        codes[:, 3::4] = (packed >> 6) & 0x3
        return _DECODE_TABLE[codes]
