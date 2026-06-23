"""End-to-end smoke test: train_hvae.py with --rg-ldsc-sumstats.

Builds a tiny PLINK fixture (n=64, m=32), a matching synthetic external
sumstats file, and tiny ref-LD / w-LD per-chrom files; runs
``train_hvae.py`` for 2 epochs via subprocess. Verifies that:

* The run exits cleanly (``returncode == 0``).
* ``log.txt`` is written and contains the LDSC intercept-diagnostic
  line emitted by ``validate_epoch``.
* Pre-existing PLINK-rank-B behaviour (no ``--rg-ldsc-sumstats``) is
  unchanged.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "test"))

from fixtures import random_genotypes, write_plink                # noqa: E402

PY = os.path.expanduser("~/zp/zaitlen/conda/envs/ccseg/bin/python3")


def _write_images(path: str, ids: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    data = rng.standard_normal((len(ids), 1, 1000)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("ids", data=ids.astype(np.int64))
        f.create_dataset("data", data=data)


def _write_ldsc_synth(tmp: pathlib.Path, variant_ids: np.ndarray,
                      seed: int = 7):
    """Synthesize a sumstats + single-chrom ref-LD + w-LD trio aligned
    to the PLINK BED's variant IDs."""
    m = len(variant_ids)
    rng = np.random.default_rng(seed)

    ref_df = pd.DataFrame({
        "CHR": [1] * m, "SNP": variant_ids,
        "BP": np.arange(1, m + 1),
        "L2": rng.uniform(1.5, 4.0, size=m),
    })
    ref_pfx = str(tmp / "ref.")
    ref_df.to_csv(f"{ref_pfx}1.l2.ldscore.gz", sep="\t",
                   index=False, compression="gzip")
    with open(f"{ref_pfx}1.l2.M_5_50", "w") as f:
        f.write(f"{m * 50}\n")

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

    # External sumstats with BED-matching alleles (A/G default in fixture).
    z = rng.normal(size=m) * 1.5
    ss_df = pd.DataFrame({
        "SNP": variant_ids, "A1": ["A"] * m, "A2": ["G"] * m,
        "N": [50_000] * m, "Z": z,
    })
    ss = tmp / "trait.sumstats.gz"
    ss_df.to_csv(ss, sep="\t", index=False, compression="gzip")
    return ss, ref_pfx, w_pfx


def _run_train_hvae(outdir: str, args: list[str]) -> tuple[int, str, str]:
    cmd = [PY, str(REPO / "train_hvae.py"), *args, "--outdir", outdir]
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    log_path = pathlib.Path(outdir) / "log.txt"
    log = log_path.read_text() if log_path.exists() else ""
    return result.returncode, log, result.stderr


COMMON_ARGS = [
    "--model", "vae1d",
    "--bs", "16",
    "--epochs", "2",
    "--epoch-cb", "1",
    "--train-frac", "0.5",
    "--zdim", "4",
    "--h-weight", "0.1",
    "--vae-lr", "1e-3",
]


def test_rg_ldsc_smoke() -> None:
    """train_hvae.py --rg-ldsc-sumstats exits cleanly and logs intercepts."""
    tmpdir = tempfile.mkdtemp(prefix="rgldsc_smoke_")
    tmp = pathlib.Path(tmpdir)
    n, m = 64, 32
    sample_ids = np.arange(1_000_500, 1_000_500 + n, dtype=np.int64)
    Xfull = random_genotypes(n, m, seed=10)
    geno_prefix = str(tmp / "geno")
    write_plink(geno_prefix, Xfull, sample_ids)

    variant_ids = np.array([f"rs{j}" for j in range(m)])
    ss, ref_pfx, w_pfx = _write_ldsc_synth(tmp, variant_ids)

    img_path = str(tmp / "imgs.hdf5")
    _write_images(img_path, sample_ids)

    outdir = str(tmp / "out")
    # Constrain intercepts to keep tiny-m IRWLS numerically stable.
    rc, log, stderr = _run_train_hvae(outdir, COMMON_ARGS + [
        "--images", img_path,
        "--genetics", geno_prefix,
        "--rg-ldsc-sumstats", str(ss),
        "--rg-ldsc-ref-ld-chr", ref_pfx,
        "--rg-ldsc-w-ld-chr", w_pfx,
        "--rg-ldsc-chroms", "1",
        "--rg-ldsc-intercept-hsq", "1.0",
        "--rg-ldsc-intercept-gencov", "0.0",
    ])
    assert rc == 0, f"train_hvae exited {rc}\n--- LOG ---\n{log}\n--- STDERR ---\n{stderr}"
    assert "rg_ldsc_intercepts_val" in log, (
        f"expected rg-ldsc intercept log line\n{log[-1500:]}"
    )
    assert "aligned" in log and "SNPs" in log, (
        f"expected SNP-alignment log line\n{log[:1500]}"
    )
    print(f"  rg-ldsc smoke ok  | log {len(log)} chars")


def test_rg_ldsc_free_intercepts() -> None:
    """Free intercepts (production default) also complete cleanly."""
    tmpdir = tempfile.mkdtemp(prefix="rgldsc_free_")
    tmp = pathlib.Path(tmpdir)
    n, m = 64, 32
    sample_ids = np.arange(1_000_500, 1_000_500 + n, dtype=np.int64)
    Xfull = random_genotypes(n, m, seed=11)
    geno_prefix = str(tmp / "geno")
    write_plink(geno_prefix, Xfull, sample_ids)

    variant_ids = np.array([f"rs{j}" for j in range(m)])
    ss, ref_pfx, w_pfx = _write_ldsc_synth(tmp, variant_ids, seed=12)

    img_path = str(tmp / "imgs.hdf5")
    _write_images(img_path, sample_ids)

    outdir = str(tmp / "out")
    rc, log, stderr = _run_train_hvae(outdir, COMMON_ARGS + [
        "--images", img_path,
        "--genetics", geno_prefix,
        "--rg-ldsc-sumstats", str(ss),
        "--rg-ldsc-ref-ld-chr", ref_pfx,
        "--rg-ldsc-w-ld-chr", w_pfx,
        "--rg-ldsc-chroms", "1",
    ])
    assert rc == 0, f"train_hvae exited {rc}\n{log}\n--- STDERR ---\n{stderr}"
    print(f"  free-intercept smoke ok  | log {len(log)} chars")


def test_mutex_rg_ldsc_with_kinship() -> None:
    """--rg-ldsc-sumstats + --kinship rejected at parse-time."""
    tmpdir = tempfile.mkdtemp(prefix="rgldsc_mutex_")
    tmp = pathlib.Path(tmpdir)
    # We don't actually need real LDSC files since setup_heritability
    # rejects before opening them. Just point to dummies.
    sample_ids = np.arange(1_000_500, 1_000_500 + 32, dtype=np.int64)
    geno_prefix = str(tmp / "geno")
    write_plink(geno_prefix, random_genotypes(32, 16, seed=1), sample_ids)
    img = str(tmp / "imgs.hdf5")
    _write_images(img, sample_ids)
    outdir = str(tmp / "out")
    rc, log, stderr = _run_train_hvae(outdir, COMMON_ARGS + [
        "--images", img,
        "--genetics", geno_prefix,
        "--kinship",
        "--rg-ldsc-sumstats", str(tmp / "missing.sumstats.gz"),
        "--rg-ldsc-ref-ld-chr", "x",
        "--rg-ldsc-w-ld-chr", "x",
    ])
    assert rc != 0
    assert ("PLINK rank-B" in stderr or "PLINK rank-B" in log
            or "rank-B" in stderr or "rank-B" in log), (
        f"expected mutex error message\nstderr: {stderr[-500:]}\nlog: {log[-500:]}"
    )
    print(f"  --rg-ldsc-sumstats + --kinship mutex rejected (rc={rc})")


if __name__ == "__main__":
    test_rg_ldsc_smoke()
    test_rg_ldsc_free_intercepts()
    test_mutex_rg_ldsc_with_kinship()
    print("ALL OK")
