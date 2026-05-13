"""load_data integration test for the PLINK inner-join branch."""
from __future__ import annotations

import pathlib
import sys
import tempfile

import h5py
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "test"))

from h2vae.data import load_data                          # noqa: E402
from fixtures import random_genotypes, write_plink         # noqa: E402


def _write_images_hdf5(path: str, ids: np.ndarray) -> None:
    n = len(ids)
    data = np.random.default_rng(42).standard_normal((n, 1, 1000)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("ids", data=ids.astype(np.int64))
        f.create_dataset("data", data=data)


def test_plink_inner_join() -> None:
    tmp = tempfile.mkdtemp(prefix="plink_load_")
    sample_ids = np.arange(1_000_100, 1_000_120, dtype=np.int64)  # n=20
    X = random_genotypes(n=20, m=8, seed=1)
    prefix = str(pathlib.Path(tmp) / "geno")
    write_plink(prefix, X, sample_ids)

    # Images cover 15 of the 20 PLINK samples (plus 3 not in PLINK).
    img_ids = np.concatenate([sample_ids[:15],
                              np.array([2_000_001, 2_000_002, 2_000_003])])
    img_path = str(pathlib.Path(tmp) / "imgs.hdf5")
    _write_images_hdf5(img_path, img_ids)

    d = load_data(
        images_path=img_path,
        genetics_path=prefix,                # PLINK prefix, no extension
        train_frac=0.6,
        seed=0,
    )

    # Inner join: img_ids ∩ plink_sample_ids = sample_ids[:15], so n=15.
    assert d["n_train"] + len(d["val_ids_raw"]) == 15
    assert d["genetics"]["plink_prefix"] == prefix
    row_idx = d["genetics"]["plink_row_idx"]
    assert row_idx is not None
    assert d["genetics"]["kinship"] is None
    assert d["genetics"]["genotypes"] is None

    # plink_row_idx should map the joined cohort (train then val) back
    # to BED row numbers; the BED row IDs at those positions should
    # match the merged ID order.
    merged = np.concatenate([d["train_ids_raw"], d["val_ids_raw"]])
    assert np.array_equal(sample_ids[row_idx], merged), (
        f"row_idx remapping incorrect:\n  bed[row_idx] = {sample_ids[row_idx]}\n"
        f"  merged       = {merged}"
    )
    print(f"  PLINK inner-join ok  (n={len(merged)}, plink_n_total=20)")


if __name__ == "__main__":
    print("load_data PLINK tests:")
    test_plink_inner_join()
    print("all tests passed.")
