"""Bit-packed sample-major in-memory cohort × variant genotype cache.

Built once at ``RankBHeritability`` construction (piggybacking on the
variant-stats BED pass), this cache replaces all subsequent BED reads.
Per the analysis in ``notes/rank_b_heritability_perf.md`` (combined
options 5+6+8), it brings the rank-B path's per-fit BED traffic in
line with SCORE's Round 1: **one disk pass, ever**.

Build is either:

* slow path: ``build_chunk(j_lo, j_hi, X_int8)`` from a numpy chunk
  decode (kept for tests and as a fallback);
* fast path: ``build_chunk_from_bed(bed_block, row_idx, j_lo, ...)``
  via the C kernel in ``_bed_pack.c`` (matches SCORE's fused inner
  loop — decode + cohort subset + bit-pack + per-variant stats in
  one pass with no intermediate (n_var × n_total) array).

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

import ctypes
import pathlib

import numpy as np

# Code → genotype lookup. Indexed by 2-bit code value.
_DECODE_TABLE = np.array([0, 1, 2, -1], dtype=np.int8)


# --------------------------------------------------------------------------
# C extension: fused BED → cache + per-variant stats
# --------------------------------------------------------------------------

_LIB_PATH = pathlib.Path(__file__).with_name("_bed_pack.so")
_LIB = None
if _LIB_PATH.exists():
    _LIB = ctypes.CDLL(str(_LIB_PATH))
    _LIB.bed_decode_and_pack.restype = None
    _LIB.bed_decode_and_pack.argtypes = [
        ctypes.c_void_p,                                    # bed_block
        ctypes.c_int64,                                     # n_var
        ctypes.c_int64,                                     # bytes_per_var_BED
        ctypes.c_void_p,                                    # row_idx (int64*)
        ctypes.c_int64,                                     # n_cohort
        ctypes.c_void_p,                                    # cache_buf
        ctypes.c_int64,                                     # bytes_per_sample
        ctypes.c_int64,                                     # j_lo
        ctypes.c_void_p,                                    # sum_x  (int64*)
        ctypes.c_void_p,                                    # sum_x2 (int64*)
        ctypes.c_void_p,                                    # n_obs  (int64*)
    ]
    _LIB.bed_decode_to_variant_int8.restype = None
    _LIB.bed_decode_to_variant_int8.argtypes = [
        ctypes.c_void_p,                                    # bed_block
        ctypes.c_int64,                                     # n_var
        ctypes.c_int64,                                     # bytes_per_var_BED
        ctypes.c_void_p,                                    # row_idx (int64*)
        ctypes.c_int64,                                     # n_cohort
        ctypes.c_void_p,                                    # out_int8
        ctypes.c_void_p,                                    # sum_x
        ctypes.c_void_p,                                    # sum_x2
        ctypes.c_void_p,                                    # n_obs
    ]
    _LIB.transpose_int8_to_bitpacked.restype = None
    _LIB.transpose_int8_to_bitpacked.argtypes = [
        ctypes.c_void_p,                                    # X_var
        ctypes.c_int64,                                     # m
        ctypes.c_int64,                                     # n_cohort
        ctypes.c_void_p,                                    # cache_out
        ctypes.c_int64,                                     # bytes_per_sample
    ]


def have_fast_kernel() -> bool:
    return _LIB is not None


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

    # ------------------------------------------------------------------
    # Fast build path: fused C kernel
    # ------------------------------------------------------------------

    def build_via_variant_major(
        self,
        bed,
        row_idx: np.ndarray,
        chunk_variants: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast cache build: BED → variant-major int8 → sample-major bit-packed.

        Steps (all in the C kernel):

        1. Stream BED variant chunks via ``bed_decode_to_variant_int8``
           into a single ``(m × n_cohort)`` int8 buffer.  Per chunk this
           is bandwidth-bound (sequential ``n_cohort``-byte writes per
           variant); no TLB / cache thrash.
        2. Per-variant ``sum``, ``sum²``, and ``n_obs`` accumulators are
           filled in the same pass — returned to the caller for stats.
        3. Tiled C-kernel transpose of the variant-major int8 buffer
           into the final sample-major bit-packed cache.

        Memory peak (during build): the variant-major int8 buffer
        (``m × n_cohort`` bytes) plus this cache (``n × ceil(m/4)``).
        After return the variant-major buffer is freed.

        Args:
            bed: ``BedFile`` open on the source PLINK.
            row_idx: ``(n_cohort,)`` BED row indices.
            chunk_variants: variant chunk size for the BED decode pass.

        Returns:
            ``(sum_x, sum_x2, n_obs)`` int64 arrays of length ``m``.
        """
        if _LIB is None:
            raise RuntimeError(
                "C kernel not built; rebuild h2vae/_bed_pack.so"
            )
        if self._built:
            raise RuntimeError("CohortCache already finalised")
        if int(bed.m) != self.m:
            raise ValueError(f"bed.m={bed.m} != cache.m={self.m}")
        n_cohort = len(row_idx)
        if n_cohort != self.n:
            raise ValueError(f"row_idx len {n_cohort} != cohort {self.n}")
        row_idx = np.ascontiguousarray(row_idx, dtype=np.int64)

        # --- Variant-major int8 buffer (large; freed before return) ---
        X_var = np.zeros((self.m, self.n), dtype=np.int8)

        sum_x_full = np.zeros(self.m, dtype=np.int64)
        sum_x2_full = np.zeros(self.m, dtype=np.int64)
        n_obs_full = np.zeros(self.m, dtype=np.int64)

        # --- Phase 1: BED → variant-major int8 (chunked) ---
        for j_lo in range(0, self.m, chunk_variants):
            j_hi = min(j_lo + chunk_variants, self.m)
            n_var = j_hi - j_lo
            bed_block = bed._mm[j_lo:j_hi]
            sum_x = np.zeros(n_var, dtype=np.int64)
            sum_x2 = np.zeros(n_var, dtype=np.int64)
            n_obs = np.zeros(n_var, dtype=np.int64)
            _LIB.bed_decode_to_variant_int8(
                bed_block.ctypes.data,
                np.int64(n_var),
                np.int64(bed.bytes_per_variant),
                row_idx.ctypes.data,
                np.int64(self.n),
                X_var[j_lo:j_hi].ctypes.data,
                sum_x.ctypes.data,
                sum_x2.ctypes.data,
                n_obs.ctypes.data,
            )
            sum_x_full[j_lo:j_hi] = sum_x
            sum_x2_full[j_lo:j_hi] = sum_x2
            n_obs_full[j_lo:j_hi] = n_obs

        # --- Phase 2: tiled transpose-and-pack into sample-major bit-packed ---
        _LIB.transpose_int8_to_bitpacked(
            X_var.ctypes.data,
            np.int64(self.m),
            np.int64(self.n),
            self._cache.ctypes.data,
            np.int64(self.bytes_per_sample),
        )

        # Free the variant-major buffer eagerly.
        del X_var
        self._built = True
        return sum_x_full, sum_x2_full, n_obs_full

    def build_chunk_from_bed(
        self,
        bed_block: np.ndarray,
        row_idx: np.ndarray,
        j_lo: int,
        sum_x: np.ndarray,
        sum_x2: np.ndarray,
        n_obs: np.ndarray,
    ) -> None:
        """Fused decode + pack + variant stats via the C kernel.

        Args:
            bed_block: ``(n_var, bytes_per_variant_BED)`` uint8 view of
                the BED mmap covering variants ``[j_lo, j_lo + n_var)``.
            row_idx: ``(n_cohort,)`` int64 BED row indices.
            j_lo: starting variant index (multiple of 4).
            sum_x, sum_x2, n_obs: per-variant int64 output buffers of
                length ``n_var``.  Caller pre-allocates and slices.
        """
        if _LIB is None:
            raise RuntimeError(
                f"C kernel not available: build {_LIB_PATH} via "
                "`gcc -O3 -std=c99 -fPIC -shared -o _bed_pack.so _bed_pack.c` "
                "inside h2vae/."
            )
        if self._built:
            raise RuntimeError("CohortCache already finalised")
        if j_lo % 4 != 0:
            raise ValueError(f"j_lo must be 4-aligned, got {j_lo}")
        if bed_block.dtype != np.uint8 or not bed_block.flags.c_contiguous:
            bed_block = np.ascontiguousarray(bed_block, dtype=np.uint8)
        row_idx = np.ascontiguousarray(row_idx, dtype=np.int64)
        n_var = bed_block.shape[0]
        if sum_x.shape != (n_var,) or sum_x.dtype != np.int64:
            raise ValueError("sum_x must be (n_var,) int64")
        if sum_x2.shape != (n_var,) or sum_x2.dtype != np.int64:
            raise ValueError("sum_x2 must be (n_var,) int64")
        if n_obs.shape != (n_var,) or n_obs.dtype != np.int64:
            raise ValueError("n_obs must be (n_var,) int64")
        _LIB.bed_decode_and_pack(
            bed_block.ctypes.data,
            np.int64(n_var),
            np.int64(bed_block.shape[1]),
            row_idx.ctypes.data,
            np.int64(len(row_idx)),
            self._cache.ctypes.data,
            np.int64(self.bytes_per_sample),
            np.int64(j_lo),
            sum_x.ctypes.data,
            sum_x2.ctypes.data,
            n_obs.ctypes.data,
        )

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
