"""PLINK BED/BIM/FAM reader.

Memory-maps the BED file (SNP-major, 2-bit packed) and exposes two
decoders:

* :meth:`BedFile.decode_variants` — for streaming rebuild
  (one variant block at a time, full cohort).
* :meth:`BedFile.decode_rows` — for rank-B updates (B sample rows
  across all variants, or a variant range).

The BED file is never decoded to fp32 in its entirety. The mmap is
managed by the kernel; per-call buffers are transient.

BED genotype encoding (2 bits per genotype, low bits first within a
byte; PLINK 1.07+ SNP-major layout):

==== =========================
0b00 homozygous for allele 1 (encoded as 2)
0b01 missing (encoded as -1)
0b10 heterozygous (encoded as 1)
0b11 homozygous for allele 2 (encoded as 0)
==== =========================

The "count A1 alleles" sign convention is irrelevant downstream because
we standardise each variant column to (mean=0, std=1) before any
heritability computation.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

# 2-bit code → genotype (count of allele 1). -1 is the missing sentinel.
_BED_TABLE = np.array([2, -1, 1, 0], dtype=np.int8)
_BED_MAGIC = bytes((0x6C, 0x1B, 0x01))


class BedFile:
    """mmap-backed reader for a PLINK BED/BIM/FAM trio.

    Args:
        prefix: Path prefix; the reader expects ``prefix + ".bed"``,
            ``prefix + ".bim"``, ``prefix + ".fam"`` to all exist.
    """

    def __init__(self, prefix: str | Path):
        self.prefix = str(prefix)
        bed_path = Path(f"{prefix}.bed")
        bim_path = Path(f"{prefix}.bim")
        fam_path = Path(f"{prefix}.fam")
        for p in (bed_path, bim_path, fam_path):
            if not p.exists():
                raise FileNotFoundError(f"PLINK file missing: {p}")

        self.sample_ids = self._parse_fam(fam_path)        # (n_total,)
        self.variant_ids, self.a1, self.a2 = self._parse_bim(bim_path)
        self.n_total = len(self.sample_ids)
        self.m = len(self.variant_ids)
        self.bytes_per_variant = (self.n_total + 3) // 4

        with open(bed_path, "rb") as f:
            magic = f.read(3)
        if magic != _BED_MAGIC:
            raise ValueError(
                f"{bed_path} has bad magic {magic!r}; expected SNP-major BED"
            )

        # Memory-map the genotype block as a 2-D uint8 array shaped
        # (m, bytes_per_variant).
        self._mm = np.memmap(
            bed_path,
            dtype=np.uint8,
            mode="r",
            offset=3,
            shape=(self.m, self.bytes_per_variant),
        )

    # ------------------------------------------------------------------
    # Header parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fam(path: Path) -> np.ndarray:
        """Return IID column (index 1) as int64 array."""
        ids = []
        with open(path) as f:
            for line in f:
                fields = line.split()
                if not fields:
                    continue
                ids.append(int(fields[1]))
        return np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _parse_bim(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(variant_ids, a1, a2)`` as parallel object arrays.

        BIM column order is ``chrom varid genpos bp a1 a2``.
        """
        ids: list[str] = []
        a1: list[str] = []
        a2: list[str] = []
        with open(path) as f:
            for line in f:
                fields = line.split()
                if not fields:
                    continue
                ids.append(fields[1])
                a1.append(fields[4] if len(fields) > 4 else "")
                a2.append(fields[5] if len(fields) > 5 else "")
        return (np.asarray(ids, dtype=object),
                np.asarray(a1, dtype=object),
                np.asarray(a2, dtype=object))

    # ------------------------------------------------------------------
    # Decoders
    # ------------------------------------------------------------------

    def decode_variants(self, j_lo: int, j_hi: int,
                        row_idx: np.ndarray | None = None) -> np.ndarray:
        """Decode a variant block to int8 ``(n_rows, j_hi - j_lo)``.

        Args:
            j_lo, j_hi: variant range, half-open.
            row_idx: optional sample-row subset of the BED's full
                ``n_total`` samples. When ``None``, returns all rows.

        Returns:
            ``int8`` array, shape ``(n_rows, j_hi - j_lo)``, with -1
            for missing genotypes.
        """
        if j_lo < 0 or j_hi > self.m or j_lo >= j_hi:
            raise ValueError(f"variant range [{j_lo}, {j_hi}) out of bounds")
        block = self._mm[j_lo:j_hi]                         # (n_var, bpv) view
        n_var = j_hi - j_lo
        codes = np.empty((n_var, self.bytes_per_variant * 4), dtype=np.uint8)
        codes[:, 0::4] = (block >> 0) & 0x3
        codes[:, 1::4] = (block >> 2) & 0x3
        codes[:, 2::4] = (block >> 4) & 0x3
        codes[:, 3::4] = (block >> 6) & 0x3
        decoded = _BED_TABLE[codes]                         # (n_var, padded)
        decoded = decoded[:, :self.n_total]                 # truncate padding
        if row_idx is None:
            return decoded.T                                # (n_total, n_var)
        return decoded[:, row_idx].T                        # (B, n_var)

    def decode_rows(self, row_idx: np.ndarray,
                    j_lo: int = 0, j_hi: int | None = None) -> np.ndarray:
        """Decode the requested rows across a variant range.

        Optimised path for rank-B updates: per variant, only the bytes
        spanning the requested samples are touched.

        Args:
            row_idx: ``(B,)`` int array of sample-row indices.
            j_lo: variant range start (default 0).
            j_hi: variant range end (default ``self.m``).

        Returns:
            ``int8`` array, shape ``(B, j_hi - j_lo)``.
        """
        if j_hi is None:
            j_hi = self.m
        if j_lo < 0 or j_hi > self.m or j_lo >= j_hi:
            raise ValueError(f"variant range [{j_lo}, {j_hi}) out of bounds")
        row_idx = np.asarray(row_idx, dtype=np.int64)
        if row_idx.ndim != 1:
            raise ValueError("row_idx must be 1-D")
        if (row_idx < 0).any() or (row_idx >= self.n_total).any():
            raise ValueError("row_idx out of range")

        byte_off = (row_idx // 4).astype(np.int64)           # (B,)
        shift = (2 * (row_idx % 4)).astype(np.uint8)         # (B,)

        # Gather the byte at (variant j, byte_off[i]) for j in [j_lo, j_hi)
        # and i in [0, B). Fancy indexing on the mmap returns a regular
        # ndarray (copy); shape (j_hi - j_lo, B).
        gathered = self._mm[j_lo:j_hi][:, byte_off]
        codes = (gathered >> shift[None, :]) & 0x3
        decoded = _BED_TABLE[codes]                          # (n_var, B)
        return decoded.T                                     # (B, n_var)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"BedFile(prefix={self.prefix!r}, n={self.n_total}, "
                f"m={self.m})")


def is_plink_prefix(path: str | Path) -> bool:
    """True iff ``<path>.bed``, ``.bim``, ``.fam`` all exist."""
    p = str(path)
    return all(Path(f"{p}{ext}").exists() for ext in (".bed", ".bim", ".fam"))


def read_fam_ids(prefix: str | Path) -> np.ndarray:
    """Light-weight helper: just the IID column from ``<prefix>.fam``.

    Used by ``load_data`` to pull the cohort for the inner-join without
    mmapping the BED itself.
    """
    return BedFile._parse_fam(Path(f"{prefix}.fam"))
