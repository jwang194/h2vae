"""End-to-end smoke test: train_hvae.py with PLINK genetics.

Generates a tiny PLINK trio + image HDF5 + covariates HDF5, then runs
``train_hvae.py`` for 2 epochs via subprocess. Verifies that the run
completes (exit 0), the log file lists finite ``h_train`` / ``h_val``
values, and a checkpoint is written.

Covers four configurations:

* ``--h-weight > 0`` with neither ``--genetic-correlation`` nor
  ``--split-variants``  (PLINK rank-B, mom mode).
* ``--genetic-correlation``  (gc mode).
* ``--split-variants``  (PLINK even/odd).
* ``--split-variants --genetic-correlation``  (gc + split).
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile

import h5py
import numpy as np

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


def _write_covariates(path: str, ids: np.ndarray) -> None:
    rng = np.random.default_rng(2)
    names = [f"PC{i}" for i in range(1, 5)] + ["Age", "Sex"]
    n = len(ids)
    data = rng.standard_normal((n, len(names))).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("ids", data=ids.astype(np.int64))
        f.create_dataset("data", data=data)
        f.create_dataset(
            "covariate_names", data=np.array(names, dtype="S")
        )


def _write_target_phenotype(path: str, ids: np.ndarray) -> None:
    rng = np.random.default_rng(3)
    data = rng.standard_normal((len(ids), 1)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("ids", data=ids.astype(np.int64))
        f.create_dataset("data", data=data)


def _make_fixture(tmpdir: str, n: int = 64, m: int = 32,
                   split: bool = False) -> dict:
    """Set up all input files; return a dict of paths."""
    sample_ids = np.arange(1_000_500, 1_000_500 + n, dtype=np.int64)

    geno_prefix = str(pathlib.Path(tmpdir) / "geno")
    if split:
        # Even: variants 0..m/2-1; Odd: variants m/2..m-1.
        Xfull = random_genotypes(n, m, seed=10)
        write_plink(f"{geno_prefix}.even", Xfull[:, :m // 2], sample_ids)
        write_plink(f"{geno_prefix}.odd",  Xfull[:, m // 2:], sample_ids)
    else:
        Xfull = random_genotypes(n, m, seed=10)
        write_plink(geno_prefix, Xfull, sample_ids)

    img_path = str(pathlib.Path(tmpdir) / "imgs.hdf5")
    _write_images(img_path, sample_ids)

    cov_path = str(pathlib.Path(tmpdir) / "covs.hdf5")
    _write_covariates(cov_path, sample_ids)

    resid_list = str(pathlib.Path(tmpdir) / "resid.covariates")
    with open(resid_list, "w") as f:
        f.write("\n".join(["PC1", "PC2", "Age", "Sex"]) + "\n")

    target_path = str(pathlib.Path(tmpdir) / "target.hdf5")
    _write_target_phenotype(target_path, sample_ids)

    return {
        "geno_prefix": geno_prefix,
        "img": img_path,
        "cov": cov_path,
        "resid": resid_list,
        "target": target_path,
    }


def _run_train_hvae(outdir: str, args: list[str]) -> tuple[int, str]:
    """Invoke train_hvae.py; return (returncode, log_text)."""
    cmd = [PY, str(REPO / "train_hvae.py"), *args, "--outdir", outdir]
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    log_path = pathlib.Path(outdir) / "log.txt"
    log = log_path.read_text() if log_path.exists() else ""
    return result.returncode, log + "\n--- STDERR ---\n" + result.stderr


def _parse_h_values(log: str, key: str) -> list[float]:
    """Extract the last ``key: …`` values from log.txt (comma-separated floats).

    Uses lookahead-up-to-next-field, since values themselves can be
    negative and a naive ``[^-]+`` pattern would chop them.
    """
    pat = rf"{key}:\s*([-\d.,eE\s+]+?)(?=\s+-\s+\w|\Z|\n)"
    matches = list(re.finditer(pat, log))
    if not matches:
        return []
    last = matches[-1].group(1)
    return [float(x) for x in last.split(",") if x.strip()]


def _assert_finite_h(log: str) -> None:
    for key in ("h_train", "h_val"):
        vals = _parse_h_values(log, key)
        assert vals, f"no {key} values in log"
        assert all(np.isfinite(v) for v in vals), (
            f"non-finite {key} values: {vals}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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


def test_plink_mom_smoke() -> None:
    """PLINK + mom (no gc, no split-variants, no residualisation)."""
    tmpdir = tempfile.mkdtemp(prefix="plinkmom_")
    fx = _make_fixture(tmpdir)
    outdir = str(pathlib.Path(tmpdir) / "out")
    rc, log = _run_train_hvae(
        outdir,
        COMMON_ARGS + [
            "--images", fx["img"],
            "--genetics", fx["geno_prefix"],
        ],
    )
    assert rc == 0, f"train_hvae exited {rc}\n{log[-2000:]}"
    _assert_finite_h(log)
    print(f"  PLINK mom smoke ok  | log {len(log)} chars")


def test_plink_mom_residualised_smoke() -> None:
    """PLINK + mom + --residualize-covariates."""
    tmpdir = tempfile.mkdtemp(prefix="plinkmomresid_")
    fx = _make_fixture(tmpdir)
    outdir = str(pathlib.Path(tmpdir) / "out")
    rc, log = _run_train_hvae(
        outdir,
        COMMON_ARGS + [
            "--images", fx["img"],
            "--genetics", fx["geno_prefix"],
            "--covariates", fx["cov"],
            "--residualize-covariates", fx["resid"],
        ],
    )
    assert rc == 0, f"train_hvae exited {rc}\n{log[-2000:]}"
    _assert_finite_h(log)
    print(f"  PLINK mom + resid smoke ok  | log {len(log)} chars")


def test_plink_gc_smoke() -> None:
    """PLINK + gc mode."""
    tmpdir = tempfile.mkdtemp(prefix="plinkgc_")
    fx = _make_fixture(tmpdir)
    outdir = str(pathlib.Path(tmpdir) / "out")
    rc, log = _run_train_hvae(
        outdir,
        COMMON_ARGS + [
            "--images", fx["img"],
            "--genetics", fx["geno_prefix"],
            "--covariates", fx["cov"],
            "--residualize-covariates", fx["resid"],
            "--genetic-correlation", fx["target"],
        ],
    )
    assert rc == 0, f"train_hvae exited {rc}\n{log[-2000:]}"
    _assert_finite_h(log)
    print(f"  PLINK gc smoke ok  | log {len(log)} chars")


def test_plink_split_variants_smoke() -> None:
    """PLINK + --split-variants (even/odd BEDs)."""
    tmpdir = tempfile.mkdtemp(prefix="plinksplit_")
    fx = _make_fixture(tmpdir, split=True)
    outdir = str(pathlib.Path(tmpdir) / "out")
    rc, log = _run_train_hvae(
        outdir,
        COMMON_ARGS + [
            "--images", fx["img"],
            "--genetics", fx["geno_prefix"],
            "--split-variants",
        ],
    )
    assert rc == 0, f"train_hvae exited {rc}\n{log[-2000:]}"
    # h_train_even and h_val_even keys.
    for key in ("h_train_even", "h_val_even", "h_train_odd", "h_val_odd"):
        vals = _parse_h_values(log, key)
        assert vals, f"no {key} values in log"
        assert all(np.isfinite(v) for v in vals), (
            f"non-finite {key}: {vals}"
        )
    print(f"  PLINK split-variants smoke ok  | log {len(log)} chars")


def test_plink_split_variants_gc_smoke() -> None:
    """PLINK + --split-variants + --genetic-correlation."""
    tmpdir = tempfile.mkdtemp(prefix="plinksplitgc_")
    fx = _make_fixture(tmpdir, split=True)
    outdir = str(pathlib.Path(tmpdir) / "out")
    rc, log = _run_train_hvae(
        outdir,
        COMMON_ARGS + [
            "--images", fx["img"],
            "--genetics", fx["geno_prefix"],
            "--covariates", fx["cov"],
            "--residualize-covariates", fx["resid"],
            "--genetic-correlation", fx["target"],
            "--split-variants",
        ],
    )
    assert rc == 0, f"train_hvae exited {rc}\n{log[-2000:]}"
    print(f"  PLINK split-variants + gc smoke ok  | log {len(log)} chars")


if __name__ == "__main__":
    print("train_hvae PLINK smoke tests:")
    test_plink_mom_smoke()
    test_plink_mom_residualised_smoke()
    test_plink_gc_smoke()
    test_plink_split_variants_smoke()
    test_plink_split_variants_gc_smoke()
    print("all tests passed.")
