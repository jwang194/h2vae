"""Loaders that align external LDSC sumstats + reference LD scores to a
``BedFile``'s variant axis, producing an ``LDSCContext`` ready to feed
to ``h2vae.ldsc_torch.RG``.

Conventions follow the original LDSC pipeline (``ldsc.py`` flags
``--ref-ld-chr``, ``--w-ld-chr``, ``--h2 sumstats.gz``):

* ``<prefix><chr>.l2.ldscore.gz`` — tab-separated, columns ``CHR SNP
  BP <annot_1> ... <annot_K>``.
* ``<prefix><chr>.l2.M_5_50`` — whitespace-separated row of length
  ``K`` (the SNP count per annotation, MAF≥5%).
* ``*.sumstats.gz`` — tab-separated, columns ``SNP A1 A2 N Z`` (LDSC
  munge output).

All paths use the same ``<prefix>`` convention as LDSC: the per-chrom
filename is built by ``f"{prefix}{chr}{suffix}"``, so ``prefix`` is
expected to include a trailing ``.`` or ``/``.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def load_sumstats(path: str | Path) -> pd.DataFrame:
    """Read an LDSC-munged sumstats file (``SNP A1 A2 N Z``)."""
    df = pd.read_csv(path, sep="\t")
    required = {"SNP", "N", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sumstats {path} missing columns: {missing}")
    return df


def load_ldscore_chr(prefix: str | Path,
                     chroms: Sequence[int] = tuple(range(1, 23)),
                     ) -> tuple[pd.DataFrame, np.ndarray]:
    """Read ``<prefix><chr>.l2.ldscore.gz`` + ``.l2.M_5_50`` across chroms.

    Returns:
        df: concatenated per-chrom LD-score table with columns
            ``CHR SNP BP <annot_1> ... <annot_K>``.
        M: ``(1, K)`` ndarray, MAF-5-50 SNP counts summed across chroms.
    """
    prefix = str(prefix)
    dfs: list[pd.DataFrame] = []
    Ms: list[np.ndarray] = []
    for c in chroms:
        df = pd.read_csv(f"{prefix}{c}.l2.ldscore.gz", sep="\t")
        with open(f"{prefix}{c}.l2.M_5_50") as f:
            m_vals = np.array([float(x) for x in f.read().split()])
        dfs.append(df)
        Ms.append(m_vals)
    df_all = pd.concat(dfs, ignore_index=True)
    M = np.stack(Ms, axis=0).sum(axis=0, keepdims=True)
    return df_all, M


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------

@dataclass
class LDSCContext:
    """Aligned per-SNP arrays ready for ``ldsc_torch.RG``.

    All tensors are double-precision; cast to the working dtype at the
    point of consumption.  Length ``m_use`` is the inner-join size
    across (sumstats, ref_ld, w_ld, bed).
    """
    ref_ld: Tensor              # (m_use, n_annot)
    w_ld: Tensor                # (m_use, 1)
    M: Tensor                   # (1, n_annot)
    z_external: Tensor          # (m_use, 1)
    n_external: Tensor          # (m_use, 1)
    bed_to_ldsc_idx: np.ndarray  # (m_use,) int, indices into the BED variant axis
    n_total_input: int           # full sumstats row count, for logging
    snp_ids: np.ndarray          # (m_use,) variant rsid strings, for diagnostics

    @property
    def m_use(self) -> int:
        return int(self.ref_ld.shape[0])

    @property
    def n_annot(self) -> int:
        return int(self.ref_ld.shape[1])

    def to(self, device, dtype=torch.float64) -> "LDSCContext":
        return LDSCContext(
            ref_ld=self.ref_ld.to(device=device, dtype=dtype),
            w_ld=self.w_ld.to(device=device, dtype=dtype),
            M=self.M.to(device=device, dtype=dtype),
            z_external=self.z_external.to(device=device, dtype=dtype),
            n_external=self.n_external.to(device=device, dtype=dtype),
            bed_to_ldsc_idx=self.bed_to_ldsc_idx,
            n_total_input=self.n_total_input,
            snp_ids=self.snp_ids,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_LDSC_NON_ANNOT_COLS = ("CHR", "SNP", "BP", "CM", "MAF")


def _annot_cols(df: pd.DataFrame) -> list[str]:
    """LD-score annotation columns.

    Excludes the LDSC-standard locus metadata columns ``CHR``/``SNP``/``BP``
    plus the optional ``CM``/``MAF`` columns that ship with the
    ``eur_w_ld_chr`` reference panel.  Everything else is treated as an
    annotation L2-score column.
    """
    return [c for c in df.columns if c not in _LDSC_NON_ANNOT_COLS]


def build_ldsc_context(
    sumstats_path: str | Path,
    ref_ld_prefix: str | Path,
    w_ld_prefix: str | Path,
    bed_variant_ids: np.ndarray,
    bed_a1: Optional[np.ndarray] = None,
    bed_a2: Optional[np.ndarray] = None,
    chroms: Sequence[int] = tuple(range(1, 23)),
) -> LDSCContext:
    """Inner-join external sumstats × ref-LD × w-LD × BED variants on SNP.

    Args:
        sumstats_path: path to a munged ``.sumstats.gz``.
        ref_ld_prefix: per-chrom LD-score prefix (e.g.
            ``"<dir>/baselineLD."``); ``<prefix><chr>.l2.ldscore.gz`` and
            ``.l2.M_5_50`` are read for each chrom.
        w_ld_prefix: same format, used for regression weights; must
            have exactly one annotation column.
        bed_variant_ids: ``(m_bed,)`` variant rsid strings in BED order.
        bed_a1, bed_a2: optional ``(m_bed,)`` allele arrays.  When
            provided, external Z is flipped where ``sumstats.A1 !=
            bed.A1`` (case-insensitive); SNPs whose alleles are
            ambiguous wrt the BED's are dropped.
        chroms: which chromosomes to load.

    Returns:
        An ``LDSCContext`` whose ``bed_to_ldsc_idx`` selects the
        per-SNP rows of the rank-B sumstats that correspond to each
        retained LDSC SNP, in the same order.
    """
    sumstats = load_sumstats(sumstats_path)
    n_in = len(sumstats)
    ref_df, ref_M = load_ldscore_chr(ref_ld_prefix, chroms)
    w_df, _ = load_ldscore_chr(w_ld_prefix, chroms)

    w_cols = _annot_cols(w_df)
    if len(w_cols) != 1:
        raise ValueError(
            f"w_ld must have exactly one annotation column; got {w_cols}"
        )
    w_df = w_df[["SNP", w_cols[0]]].rename(columns={w_cols[0]: "W_L2"})

    ref_cols = _annot_cols(ref_df)
    ref_df_lean = ref_df[["SNP"] + ref_cols]

    bed_df = pd.DataFrame({
        "SNP": np.asarray(bed_variant_ids),
        "bed_idx": np.arange(len(bed_variant_ids), dtype=np.int64),
    })
    if bed_a1 is not None:
        bed_df["BED_A1"] = np.asarray(bed_a1)
        bed_df["BED_A2"] = np.asarray(bed_a2 if bed_a2 is not None else bed_a1)

    df = sumstats.merge(ref_df_lean, on="SNP", how="inner")
    df = df.merge(w_df, on="SNP", how="inner")
    df = df.merge(bed_df, on="SNP", how="inner")

    # Filter ambiguous-allele rows up front, then sort, then apply flip.
    if bed_a1 is not None and "A1" in df.columns and "A2" in df.columns:
        ss_a1 = df["A1"].str.upper().to_numpy()
        ss_a2 = df["A2"].str.upper().to_numpy()
        b_a1 = df["BED_A1"].astype(str).str.upper().to_numpy()
        b_a2 = df["BED_A2"].astype(str).str.upper().to_numpy()
        match = (ss_a1 == b_a1) & (ss_a2 == b_a2)
        flip = (ss_a1 == b_a2) & (ss_a2 == b_a1)
        df = df.loc[match | flip].reset_index(drop=True)

    df = df.sort_values("bed_idx").reset_index(drop=True)
    z = df["Z"].to_numpy(dtype=np.float64)
    if bed_a1 is not None and "A1" in df.columns:
        ss_a1 = df["A1"].str.upper().to_numpy()
        b_a1 = df["BED_A1"].astype(str).str.upper().to_numpy()
        z = np.where(ss_a1 != b_a1, -z, z)

    return LDSCContext(
        ref_ld=torch.from_numpy(df[ref_cols].to_numpy(dtype=np.float64)),
        w_ld=torch.from_numpy(df[["W_L2"]].to_numpy(dtype=np.float64)),
        M=torch.from_numpy(ref_M.astype(np.float64)),
        z_external=torch.from_numpy(z[:, None]),
        n_external=torch.from_numpy(df["N"].to_numpy(dtype=np.float64)[:, None]),
        bed_to_ldsc_idx=df["bed_idx"].to_numpy(),
        n_total_input=int(n_in),
        snp_ids=df["SNP"].to_numpy(),
    )
