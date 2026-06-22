"""Tests for ``h2vae.ldsc_io``.

Synthetic round-trip: write tiny ``.sumstats.gz``, per-chrom
``.l2.ldscore.gz`` / ``.l2.M_5_50``, then build the LDSCContext and
verify alignment, multi-annotation handling, and allele-flip semantics.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from h2vae.ldsc_io import (                                       # noqa: E402
    LDSCContext, load_sumstats, load_ldscore_chr, build_ldsc_context,
)


# ---------------------------------------------------------------------------
# Synthetic file writers
# ---------------------------------------------------------------------------

def _write_sumstats(path: pathlib.Path, df: pd.DataFrame) -> None:
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def _write_ldscore_chr(prefix: str, chrom: int, df: pd.DataFrame,
                       M_5_50: list[float]) -> None:
    df.to_csv(f"{prefix}{chrom}.l2.ldscore.gz", sep="\t", index=False,
              compression="gzip")
    with open(f"{prefix}{chrom}.l2.M_5_50", "w") as f:
        f.write(" ".join(str(x) for x in M_5_50) + "\n")


def make_fixture(tmp: pathlib.Path):
    """One-chrom synthetic dataset.  Returns (sumstats_path, ref_pfx, w_pfx,
    bed_ids, bed_a1, bed_a2, expected_z_for_bed_ids)."""
    # SNP universe: rs0..rs9 in BED.
    bed_ids = np.array([f"rs{i}" for i in range(10)])
    bed_a1 = np.array(["A"] * 10)
    bed_a2 = np.array(["G"] * 10)

    # ref-ld covers rs0..rs7, two annots (baseL2, coding).
    ref_df = pd.DataFrame({
        "CHR": [1] * 8,
        "SNP": [f"rs{i}" for i in range(8)],
        "BP": [10 * i + 1 for i in range(8)],
        "baseL2": np.linspace(1.5, 4.0, 8),
        "codingL2": np.linspace(0.1, 0.9, 8),
    })
    ref_pfx = str(tmp / "ref.")
    _write_ldscore_chr(ref_pfx, 1, ref_df, M_5_50=[5000.0, 800.0])

    # w-ld covers rs1..rs9, single annot.
    w_df = pd.DataFrame({
        "CHR": [1] * 9,
        "SNP": [f"rs{i}" for i in range(1, 10)],
        "BP": [10 * i + 1 for i in range(1, 10)],
        "L2": np.linspace(0.7, 2.1, 9),
    })
    w_pfx = str(tmp / "w.")
    _write_ldscore_chr(w_pfx, 1, w_df, M_5_50=[6000.0])

    # Sumstats covers rs2..rs9 (so the inner-join is rs2..rs7).
    # A1='A', A2='G' agreeing with BED on most; flip rs3, rs5 to test allele
    # flipping; rs6 has ambiguous alleles (should be dropped).
    sumstats_df = pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(2, 10)],
        "A1": ["A", "G", "A", "G", "C", "A", "A", "A"],  # rs3, rs5 flipped; rs6 ambiguous
        "A2": ["G", "A", "G", "A", "T", "G", "G", "G"],
        "N": [12000] * 8,
        "Z": np.arange(2, 10, dtype=float),
    })
    ss_path = tmp / "trait.sumstats.gz"
    _write_sumstats(ss_path, sumstats_df)

    return ss_path, ref_pfx, w_pfx, bed_ids, bed_a1, bed_a2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_sumstats_columns() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        df = pd.DataFrame({"SNP": ["rs1"], "A1": ["A"], "A2": ["G"],
                           "N": [1000], "Z": [0.5]})
        p = tmp / "t.sumstats.gz"
        _write_sumstats(p, df)
        loaded = load_sumstats(p)
        assert set(["SNP", "N", "Z"]).issubset(loaded.columns)
        assert loaded.iloc[0]["Z"] == 0.5
    print("  load_sumstats OK")


def test_load_ldscore_chr_concat() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        pfx = str(tmp / "p.")
        for chrom, snps in [(1, ["rs1", "rs2"]), (2, ["rs3", "rs4", "rs5"])]:
            df = pd.DataFrame({
                "CHR": [chrom] * len(snps),
                "SNP": snps,
                "BP": list(range(len(snps))),
                "L2": [1.0] * len(snps),
            })
            _write_ldscore_chr(pfx, chrom, df, [100.0])
        df_all, M = load_ldscore_chr(pfx, chroms=[1, 2])
        assert len(df_all) == 5
        assert M.shape == (1, 1) and M[0, 0] == 200.0
    print("  load_ldscore_chr concat OK")


def test_build_context_alignment_and_flip() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        ss, ref_pfx, w_pfx, bed_ids, bed_a1, bed_a2 = make_fixture(tmp)

        ctx = build_ldsc_context(
            ss, ref_pfx, w_pfx,
            bed_variant_ids=bed_ids,
            bed_a1=bed_a1, bed_a2=bed_a2,
            chroms=[1],
        )
        # Inner-join: sumstats(rs2..rs9) ∩ ref(rs0..rs7) ∩ w(rs1..rs9) ∩ bed(rs0..rs9)
        # = rs2..rs7.  rs6 is dropped (alleles don't match BED).
        # → expected rs2, rs3, rs4, rs5, rs7
        expected_snps = ["rs2", "rs3", "rs4", "rs5", "rs7"]
        assert list(ctx.snp_ids) == expected_snps, (
            f"snps {ctx.snp_ids}"
        )
        assert ctx.m_use == 5
        assert ctx.n_annot == 2

        # bed_to_ldsc_idx must select the correct BED rows in the
        # same order as snp_ids.
        for i, rsid in enumerate(expected_snps):
            j_bed = ctx.bed_to_ldsc_idx[i]
            assert bed_ids[j_bed] == rsid

        # Z flipping: rs3 had A1=G/A2=A → flip; rs5 had A1=G/A2=A → flip;
        # rs2, rs4, rs7 had A1=A/A2=G → no flip.
        # Original Z values were [2,3,4,5,6,7,8,9] for rs2..rs9.
        # After dropping rs6 and ordering by bed_idx: rs2(2), rs3(-3), rs4(4),
        # rs5(-5), rs7(7).
        z = ctx.z_external.squeeze(-1).numpy()
        np.testing.assert_allclose(z, [2.0, -3.0, 4.0, -5.0, 7.0])

        # M is summed across chroms (single chrom here).
        assert ctx.M.shape == (1, 2)
        np.testing.assert_allclose(ctx.M.numpy(), [[5000.0, 800.0]])
    print("  build_ldsc_context alignment + flip OK")


def test_build_context_no_alleles_keeps_z() -> None:
    """When BED alleles are not provided, no flipping happens and all
    inner-joined SNPs are kept."""
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        ss, ref_pfx, w_pfx, bed_ids, _, _ = make_fixture(tmp)
        ctx = build_ldsc_context(
            ss, ref_pfx, w_pfx,
            bed_variant_ids=bed_ids, chroms=[1],
        )
        # Without allele info, rs6 is kept too: inner-join is rs2..rs7.
        assert ctx.m_use == 6
        # Z values are unchanged (no flip).
        z = ctx.z_external.squeeze(-1).numpy()
        np.testing.assert_allclose(z, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    print("  build_ldsc_context no-alleles passthrough OK")


def test_w_ld_must_be_single_annot() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        ss, ref_pfx, _, bed_ids, _, _ = make_fixture(tmp)
        # Build a multi-annot w-ld file → should error.
        bad_w = pd.DataFrame({
            "CHR": [1, 1], "SNP": ["rs2", "rs3"], "BP": [1, 2],
            "L2_a": [0.5, 0.6], "L2_b": [0.7, 0.8],
        })
        bad_pfx = str(tmp / "bad_w.")
        _write_ldscore_chr(bad_pfx, 1, bad_w, [100.0, 200.0])
        try:
            build_ldsc_context(ss, ref_pfx, bad_pfx,
                                bed_variant_ids=bed_ids, chroms=[1])
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "exactly one annotation" in str(e)
    print("  w-ld single-annotation guard OK")


if __name__ == "__main__":
    test_load_sumstats_columns()
    test_load_ldscore_chr_concat()
    test_build_context_alignment_and_flip()
    test_build_context_no_alleles_keeps_z()
    test_w_ld_must_be_single_annot()
    print("ALL OK")
